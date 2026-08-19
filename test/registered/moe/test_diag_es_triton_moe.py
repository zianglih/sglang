import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.diag_es.moe_ops import (
    apply_moe_fc2_pre_delta_inplace,
    materialize_moe_fc1_pre_input,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    fused_experts_impl,
    get_diag_es_moe_inputs,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "Diag ES Triton MoE needs CUDA")
class TestDiagEsTritonMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.cuda.set_device(0)
        cls._owns_distributed_environment = False
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=(f"tcp://127.0.0.1:{find_available_port(29500)}"),
            backend="nccl",
        )
        cls._owns_distributed_environment = True
        initialize_model_parallel(tensor_model_parallel_size=1)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_owns_distributed_environment", False):
            destroy_model_parallel()
            destroy_distributed_environment()
        super().tearDownClass()

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_get_inputs_respects_active_placement(self):
        layer_id = 7
        token_slots = object()
        runner_config = SimpleNamespace(es_layer_id=layer_id)
        placements = {
            "pre": ("moe_fc1_pre", "moe_fc2_pre"),
            "post": ("moe_fc1_post", "moe_fc2_post"),
            "both": (
                "moe_fc1_pre",
                "moe_fc1_post",
                "moe_fc2_pre",
                "moe_fc2_post",
            ),
        }

        for placement, active_names in placements.items():
            with self.subTest(placement=placement):
                banks = {name: object() for name in active_names}
                manager = SimpleNamespace(
                    manifest=SimpleNamespace(
                        grouped_gate_shapes={name: () for name in active_names}
                    ),
                    get_expert_delta_bank=Mock(
                        side_effect=lambda requested_layer, name: banks[name]
                    ),
                )
                with (
                    patch(
                        "sglang.srt.diag_es.get_diag_es_manager",
                        return_value=manager,
                    ),
                    patch(
                        "sglang.srt.model_executor.forward_context.get_forward_context",
                        return_value=SimpleNamespace(es_candidate_slots=token_slots),
                    ),
                ):
                    actual = get_diag_es_moe_inputs(runner_config)

                self.assertIs(actual[0], token_slots)
                for index, name in enumerate(
                    (
                        "moe_fc1_pre",
                        "moe_fc1_post",
                        "moe_fc2_pre",
                        "moe_fc2_post",
                    ),
                    start=1,
                ):
                    self.assertIs(actual[index], banks.get(name))
                self.assertEqual(
                    [
                        item.args
                        for item in manager.get_expert_delta_bank.call_args_list
                    ],
                    [(layer_id, name) for name in active_names],
                )

    @staticmethod
    def _torch_oracle(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        token_slots,
        fc1_pre,
        fc1_post,
        fc2_pre,
        fc2_post,
        b1=None,
        b2=None,
    ):
        num_tokens, topk = topk_ids.shape
        intermediate_size = w2.shape[-1]
        route_outputs = torch.empty(
            (num_tokens, topk, hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for token in range(num_tokens):
            slot = int(token_slots[token])
            for route in range(topk):
                expert = int(topk_ids[token, route])
                hidden_fp32 = hidden_states[token].float()
                if fc1_pre is not None:
                    fc1_input = torch.addcmul(
                        hidden_fp32, hidden_fp32, fc1_pre[expert, slot]
                    ).to(hidden_states.dtype)
                else:
                    fc1_input = hidden_states[token]
                gate_up_fp32 = fc1_input.float() @ w13[expert].float().transpose(0, 1)
                if b1 is not None:
                    gate_up_fp32 += b1[expert]
                if fc1_post is not None:
                    gate_up_fp32 = torch.addcmul(
                        gate_up_fp32, gate_up_fp32, fc1_post[expert, slot]
                    )
                gate_up = gate_up_fp32.to(hidden_states.dtype)
                activated = (
                    F.silu(gate_up[:intermediate_size]) * gate_up[intermediate_size:]
                ).to(hidden_states.dtype)
                activated_fp32 = activated.float()
                if fc2_pre is not None:
                    fc2_input = torch.addcmul(
                        activated_fp32, activated_fp32, fc2_pre[expert, slot]
                    ).to(hidden_states.dtype)
                else:
                    fc2_input = activated
                fc2_output_fp32 = fc2_input.float() @ w2[expert].float().transpose(0, 1)
                if b2 is not None:
                    fc2_output_fp32 += b2[expert]
                if fc2_post is not None:
                    fc2_output_fp32 = torch.addcmul(
                        fc2_output_fp32,
                        fc2_output_fp32,
                        fc2_post[expert, slot],
                    )
                route_outputs[token, route] = (
                    fc2_output_fp32 * topk_weights[token, route]
                ).to(hidden_states.dtype)

        return route_outputs.sum(dim=1)

    def test_mixed_candidate_expert_specific_fc1_fc2(self):
        torch.manual_seed(20260811)
        num_tokens = 11
        # Default BF16 MoE uses BLOCK_SIZE_N=64 for this M/E regime; 96-wide
        # FC1 and FC2 outputs exercise the post-delta tail mask and raw-N stride.
        hidden_size = 96
        intermediate_size = 48
        num_experts = 4
        topk = 2
        physical_slots = 3

        hidden_states = torch.randn(
            (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        w13 = (
            torch.randn(
                (num_experts, 2 * intermediate_size, hidden_size),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        ).contiguous()
        w2 = (
            torch.randn(
                (num_experts, hidden_size, intermediate_size),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        ).contiguous()
        b1 = torch.randn(
            (num_experts, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )
        b2 = torch.randn((num_experts, hidden_size), device="cuda", dtype=torch.float32)

        topk_ids = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [1, 3],
                [0, 2],
                [3, 0],
                [2, 1],
                [0, 3],
                [1, 2],
                [2, 0],
                [3, 1],
                [1, 0],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        topk_weights = torch.softmax(
            torch.randn((num_tokens, topk), device="cuda", dtype=torch.float32),
            dim=-1,
        ).contiguous()
        token_slots = torch.tensor(
            [0, 1, 2, 1, 2, 0, 2, 1, 0, 2, 1],
            device="cuda",
            dtype=torch.int32,
        )

        fc1_pre = torch.zeros(
            (num_experts, physical_slots, hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc1_post = torch.zeros(
            (num_experts, physical_slots, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc2_pre = torch.zeros(
            (num_experts, physical_slots, intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc2_post = torch.zeros(
            (num_experts, physical_slots, hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc1_pre[:, 1:].uniform_(-0.35, 0.35)
        fc1_post[:, 1:].uniform_(-0.25, 0.25)
        fc2_pre[:, 1:].uniform_(-0.55, 0.55)
        fc2_post[:, 1:].uniform_(-0.15, 0.15)

        def run(mode, enabled):
            with envs.SGLANG_DIAG_ES_MOE_GATE_MODE.override(mode):
                return fused_experts_impl(
                    hidden_states,
                    w13,
                    w2,
                    topk_weights,
                    topk_ids,
                    b1=b1,
                    b2=b2,
                    inplace=False,
                    activation="silu",
                    is_gated=True,
                    filter_expert=False,
                    diag_es_token_slots=token_slots,
                    diag_es_fc1_pre=fc1_pre if "fc1_pre" in enabled else None,
                    diag_es_fc1_post=fc1_post if "fc1_post" in enabled else None,
                    diag_es_fc2_pre=fc2_pre if "fc2_pre" in enabled else None,
                    diag_es_fc2_post=fc2_post if "fc2_post" in enabled else None,
                )

        scenarios = (
            frozenset(("fc1_pre",)),
            frozenset(("fc1_post",)),
            frozenset(("fc2_pre",)),
            frozenset(("fc2_post",)),
            frozenset(("fc1_pre", "fc1_post", "fc2_pre", "fc2_post")),
        )
        for enabled in scenarios:
            unfused = run("unfused", enabled)
            fused = run("fused", enabled)
            expected = self._torch_oracle(
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                token_slots,
                fc1_pre if "fc1_pre" in enabled else None,
                fc1_post if "fc1_post" in enabled else None,
                fc2_pre if "fc2_pre" in enabled else None,
                fc2_post if "fc2_post" in enabled else None,
                b1,
                b2,
            )

            torch.testing.assert_close(unfused, expected, rtol=0.12, atol=0.03)
            torch.testing.assert_close(fused, expected, rtol=0.12, atol=0.03)
            torch.testing.assert_close(fused, unfused, rtol=0, atol=0)

    def test_unfused_pointwise_route_layout(self):
        torch.manual_seed(20260812)
        hidden_states = torch.randn((3, 16), device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.tensor(
            [[0, 1], [2, -1], [1, 0]], device="cuda", dtype=torch.int32
        )
        token_slots = torch.tensor([0, 2, 1], device="cuda", dtype=torch.int32)
        fc1_pre = torch.randn((3, 3, 16), device="cuda", dtype=torch.float32)

        fc1_actual = materialize_moe_fc1_pre_input(
            hidden_states, topk_ids, token_slots, fc1_pre
        )
        fc1_expected = torch.empty_like(fc1_actual)
        for route, expert in enumerate(topk_ids.view(-1).tolist()):
            token = route // topk_ids.shape[1]
            if expert < 0:
                fc1_expected[route].zero_()
            else:
                hidden_fp32 = hidden_states[token].float()
                delta = fc1_pre[expert, int(token_slots[token])]
                fc1_expected[route] = torch.addcmul(hidden_fp32, hidden_fp32, delta).to(
                    hidden_states.dtype
                )
        torch.testing.assert_close(fc1_actual, fc1_expected, rtol=0, atol=0)

        fc2_actual = torch.randn(
            (topk_ids.numel(), 8), device="cuda", dtype=torch.bfloat16
        )
        fc2_pre = torch.randn((3, 3, 8), device="cuda", dtype=torch.float32)
        fc2_expected = torch.empty_like(fc2_actual)
        for route, expert in enumerate(topk_ids.view(-1).tolist()):
            token = route // topk_ids.shape[1]
            if expert < 0:
                fc2_expected[route].zero_()
            else:
                activation_fp32 = fc2_actual[route].float()
                delta = fc2_pre[expert, int(token_slots[token])]
                fc2_expected[route] = torch.addcmul(
                    activation_fp32, activation_fp32, delta
                ).to(fc2_actual.dtype)

        apply_moe_fc2_pre_delta_inplace(fc2_actual, topk_ids, token_slots, fc2_pre)
        torch.testing.assert_close(fc2_actual, fc2_expected, rtol=0, atol=0)

    def test_zero_post_delta_is_bitwise_identity(self):
        torch.manual_seed(20260820)
        num_tokens, hidden_size, intermediate_size = 7, 64, 32
        num_experts, topk, physical_slots = 3, 2, 4
        hidden_states = torch.randn(
            (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        w13 = torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        w2 = torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1], [0, 1]],
            device="cuda",
            dtype=torch.int32,
        )
        topk_weights = torch.softmax(
            torch.randn((num_tokens, topk), device="cuda", dtype=torch.float32),
            dim=-1,
        ).contiguous()
        token_slots = torch.tensor(
            [0, 1, 2, 3, 1, 2, 0], device="cuda", dtype=torch.int32
        )
        fc1_post = torch.zeros(
            (num_experts, physical_slots, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc2_post = torch.zeros(
            (num_experts, physical_slots, hidden_size),
            device="cuda",
            dtype=torch.float32,
        )

        baseline = fused_experts_impl(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            is_gated=True,
            filter_expert=False,
        )
        identity = fused_experts_impl(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            is_gated=True,
            filter_expert=False,
            diag_es_token_slots=token_slots,
            diag_es_fc1_post=fc1_post,
            diag_es_fc2_post=fc2_post,
        )

        torch.testing.assert_close(identity, baseline, rtol=0, atol=0)

    def test_post_delta_cuda_graph_observes_live_slots_and_banks(self):
        torch.manual_seed(20260821)
        num_tokens, hidden_size, intermediate_size = 5, 64, 32
        num_experts, topk, physical_slots = 3, 2, 3
        hidden_states = torch.randn(
            (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        w13 = torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        w2 = torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]],
            device="cuda",
            dtype=torch.int32,
        )
        topk_weights = torch.softmax(
            torch.randn((num_tokens, topk), device="cuda", dtype=torch.float32),
            dim=-1,
        ).contiguous()
        token_slots = torch.tensor([0, 1, 2, 1, 2], device="cuda", dtype=torch.int32)
        fc1_post = torch.zeros(
            (num_experts, physical_slots, 2 * intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc2_post = torch.zeros(
            (num_experts, physical_slots, hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc1_post[:, 1:].uniform_(-0.1, 0.1)
        fc2_post[:, 1:].uniform_(-0.1, 0.1)

        def run():
            return fused_experts_impl(
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                inplace=False,
                activation="silu",
                is_gated=True,
                no_combine=True,
                filter_expert=False,
                diag_es_token_slots=token_slots,
                diag_es_fc1_post=fc1_post,
                diag_es_fc2_post=fc2_post,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()

        graph.replay()
        torch.cuda.synchronize()
        initial_expected = run()
        torch.testing.assert_close(captured, initial_expected, rtol=0, atol=0)

        token_slots.copy_(
            torch.tensor([2, 0, 1, 2, 1], device="cuda", dtype=torch.int32)
        )
        fc1_post[:, 1:].uniform_(-0.2, 0.2)
        fc2_post[:, 1:].uniform_(-0.2, 0.2)
        graph.replay()
        torch.cuda.synchronize()
        mutated_expected = run()
        torch.testing.assert_close(captured, mutated_expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
