import unittest

import torch
import torch.nn.functional as F

from sglang.srt.diag_es.moe_ops import (
    apply_moe_fc2_delta_inplace,
    materialize_moe_fc1_input,
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

    @staticmethod
    def _torch_oracle(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        token_slots,
        fc1_delta,
        fc2_delta,
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
                fc1_input = torch.addcmul(
                    hidden_fp32, hidden_fp32, fc1_delta[expert, slot]
                ).to(hidden_states.dtype)
                gate_up = fc1_input @ w13[expert].transpose(0, 1)
                activated = (
                    F.silu(gate_up[:intermediate_size]) * gate_up[intermediate_size:]
                ).to(hidden_states.dtype)
                activated_fp32 = activated.float()
                fc2_input = torch.addcmul(
                    activated_fp32, activated_fp32, fc2_delta[expert, slot]
                ).to(hidden_states.dtype)
                route_outputs[token, route] = fc2_input @ w2[expert].transpose(0, 1)

        return (route_outputs * topk_weights.to(hidden_states.dtype).unsqueeze(-1)).sum(
            dim=1
        )

    def test_mixed_candidate_expert_specific_fc1_fc2(self):
        torch.manual_seed(20260811)
        num_tokens = 11
        hidden_size = 128
        intermediate_size = 64
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

        fc1_delta = torch.zeros(
            (num_experts, physical_slots, hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc2_delta = torch.zeros(
            (num_experts, physical_slots, intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )
        fc1_delta[:, 1:].uniform_(
            -0.35,
            0.35,
        )
        fc2_delta[:, 1:].uniform_(
            -0.55,
            0.55,
        )

        def run(mode):
            with envs.SGLANG_DIAG_ES_MOE_GATE_MODE.override(mode):
                return fused_experts_impl(
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
                    diag_es_fc1_delta=fc1_delta,
                    diag_es_fc2_delta=fc2_delta,
                )

        unfused = run("unfused")
        fused = run("fused")
        expected = self._torch_oracle(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            token_slots,
            fc1_delta,
            fc2_delta,
        )

        torch.testing.assert_close(unfused, expected, rtol=0.12, atol=0.02)
        torch.testing.assert_close(fused, expected, rtol=0.12, atol=0.02)
        torch.testing.assert_close(fused, unfused, rtol=0, atol=0)

    def test_unfused_pointwise_route_layout(self):
        torch.manual_seed(20260812)
        hidden_states = torch.randn((3, 16), device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.tensor(
            [[0, 1], [2, -1], [1, 0]], device="cuda", dtype=torch.int32
        )
        token_slots = torch.tensor([0, 2, 1], device="cuda", dtype=torch.int32)
        fc1_delta = torch.randn((3, 3, 16), device="cuda", dtype=torch.float32)

        fc1_actual = materialize_moe_fc1_input(
            hidden_states, topk_ids, token_slots, fc1_delta
        )
        fc1_expected = torch.empty_like(fc1_actual)
        for route, expert in enumerate(topk_ids.view(-1).tolist()):
            token = route // topk_ids.shape[1]
            if expert < 0:
                fc1_expected[route].zero_()
            else:
                hidden_fp32 = hidden_states[token].float()
                delta = fc1_delta[expert, int(token_slots[token])]
                fc1_expected[route] = torch.addcmul(hidden_fp32, hidden_fp32, delta).to(
                    hidden_states.dtype
                )
        torch.testing.assert_close(fc1_actual, fc1_expected, rtol=0, atol=0)

        fc2_actual = torch.randn(
            (topk_ids.numel(), 8), device="cuda", dtype=torch.bfloat16
        )
        fc2_delta = torch.randn((3, 3, 8), device="cuda", dtype=torch.float32)
        fc2_expected = torch.empty_like(fc2_actual)
        for route, expert in enumerate(topk_ids.view(-1).tolist()):
            token = route // topk_ids.shape[1]
            if expert < 0:
                fc2_expected[route].zero_()
            else:
                activation_fp32 = fc2_actual[route].float()
                delta = fc2_delta[expert, int(token_slots[token])]
                fc2_expected[route] = torch.addcmul(
                    activation_fp32, activation_fp32, delta
                ).to(fc2_actual.dtype)

        apply_moe_fc2_delta_inplace(fc2_actual, topk_ids, token_slots, fc2_delta)
        torch.testing.assert_close(fc2_actual, fc2_expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
