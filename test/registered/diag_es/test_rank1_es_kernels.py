from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rank-1 ES Triton kernels require CUDA"
)


def test_rank1_launch_config_uses_measured_row_threshold():
    from sglang.srt.diag_es.ops import rank1_launch_config

    assert rank1_launch_config(1) == (512, 512, 8)
    assert rank1_launch_config(1024) == (512, 512, 8)
    assert rank1_launch_config(1025) == (256, 256, 4)


def _dense_reference(x, base_output, down_bank, up_bank, slots):
    scalar = (x.float() * down_bank[slots.long()]).sum(dim=-1, keepdim=True)
    return (base_output.float() + scalar * up_bank[slots.long()]).bfloat16()


def _moe_reference(
    activations,
    base_output,
    topk_ids,
    topk_weights,
    slots,
    down_bank,
    up_bank,
    *,
    input_route_major,
    apply_router_weight,
):
    output = base_output.view(-1, base_output.shape[-1]).float().clone()
    topk = topk_ids.shape[1]
    for route in range(topk_ids.numel()):
        token = route // topk
        expert = int(topk_ids.view(-1)[route])
        if expert < 0:
            continue
        slot = int(slots[token])
        activation_row = route if input_route_major else token
        scalar = (
            activations.view(-1, activations.shape[-1])[activation_row].float()
            * down_bank[expert, slot]
        ).sum()
        if apply_router_weight:
            scalar *= topk_weights.view(-1)[route]
        output[route] += scalar * up_bank[expert, slot]
    return output.bfloat16().view_as(base_output)


def _moe_sorted_reference(
    activations,
    base_output,
    sorted_token_ids,
    num_tokens_post_padded,
    topk_ids,
    topk_weights,
    slots,
    down_bank,
    up_bank,
    *,
    input_sorted,
    output_sorted,
    apply_router_weight,
):
    output = base_output.view(-1, base_output.shape[-1]).float().clone()
    topk = topk_ids.shape[1]
    num_routes = topk_ids.numel()
    for sorted_position in range(int(num_tokens_post_padded[0])):
        route = int(sorted_token_ids[sorted_position])
        if route < 0 or route >= num_routes:
            continue
        token = route // topk
        expert = int(topk_ids.view(-1)[route])
        if expert < 0:
            continue
        slot = int(slots[token])
        activation_row = sorted_position if input_sorted else token
        output_row = sorted_position if output_sorted else route
        scalar = (
            activations.view(-1, activations.shape[-1])[activation_row].float()
            * down_bank[expert, slot]
        ).sum()
        if apply_router_weight:
            scalar *= topk_weights.view(-1)[route]
        output[output_row] += scalar * up_bank[expert, slot]
    return output.bfloat16().view_as(base_output)


def test_dense_rank1_mixed_slots_identity_and_caller_owned_output():
    from sglang.srt.diag_es.ops import apply_dense_rank1_out

    torch.manual_seed(20260830)
    rows, k, n = 11, 263, 389
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    base_output = torch.randn((rows, n), device="cuda", dtype=torch.bfloat16)
    down_bank = torch.randn((3, k), device="cuda", dtype=torch.float32) * 0.01
    up_bank = torch.randn((3, n), device="cuda", dtype=torch.float32) * 0.1
    down_bank[0].zero_()
    up_bank[0].zero_()
    slots = torch.tensor(
        [0, 1, 2, 0, 2, 1, 1, 0, 2, 0, 1],
        device="cuda",
        dtype=torch.int32,
    )
    output = torch.empty_like(base_output)

    returned = apply_dense_rank1_out(x, base_output, down_bank, up_bank, slots, output)
    expected = _dense_reference(x, base_output, down_bank, up_bank, slots)

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, expected, rtol=8e-3, atol=2e-2)
    assert torch.equal(output[slots == 0], base_output[slots == 0])


@pytest.mark.parametrize("rows", [1024, 1025])
def test_dense_rank1_tuned_launch_threshold_executes_both_geometries(rows):
    from sglang.srt.diag_es.ops import apply_dense_rank1_out

    torch.manual_seed(20260830 + rows)
    k, n = 263, 389
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    base_output = torch.randn((rows, n), device="cuda", dtype=torch.bfloat16)
    down_bank = torch.randn((3, k), device="cuda", dtype=torch.float32) * 0.01
    up_bank = torch.randn((3, n), device="cuda", dtype=torch.float32) * 0.1
    down_bank[0].zero_()
    up_bank[0].zero_()
    slots = torch.arange(rows, device="cuda", dtype=torch.int32) % 3
    output = torch.empty_like(base_output)

    apply_dense_rank1_out(x, base_output, down_bank, up_bank, slots, output)
    expected = _dense_reference(x, base_output, down_bank, up_bank, slots)

    torch.testing.assert_close(output, expected, rtol=8e-3, atol=2e-2)
    assert torch.equal(output[slots == 0], base_output[slots == 0])


def test_dense_rank1_fp32_contract_keeps_factor_signal_until_final_bf16_store():
    from sglang.srt.diag_es.ops import apply_dense_rank1_inplace

    x = torch.tensor(
        [[1.9921875, 1.984375, 1.5, 127.5]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    base_output = torch.tensor(
        [[1.9921875, 1.984375, 1.5, 127.5]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    down_bank = torch.full((1, 4), 0.003, device="cuda", dtype=torch.float32)
    up_bank = torch.tensor(
        [[0.25, -0.25, 0.5, -0.5]], device="cuda", dtype=torch.float32
    )
    slots = torch.zeros(1, device="cuda", dtype=torch.int32)
    original = base_output.clone()

    actual = apply_dense_rank1_inplace(x, base_output, down_bank, up_bank, slots)
    expected = _dense_reference(x, original, down_bank, up_bank, slots)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, original)


def test_bf16_linear_rank1_apply_into_is_cuda_graph_replayable(monkeypatch):
    from sglang.kernels.ops.gemm.triton_bf16_gemm import triton_bf16_linear
    from sglang.srt.layers.quantization import unquant as unquant_module
    from sglang.srt.layers.quantization.unquant import (
        Bf16GemmBackend,
        UnquantizedLinearMethod,
    )
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    monkeypatch.setattr(unquant_module, "_BF16_GEMM_BACKEND", Bf16GemmBackend.TRITON)
    torch.manual_seed(20260831)
    rows, k, n = 17, 256, 192
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1
    down_bank = torch.randn((3, k), device="cuda", dtype=torch.float32) * 0.01
    up_bank = torch.randn((3, n), device="cuda", dtype=torch.float32) * 0.1
    down_bank[0].zero_()
    up_bank[0].zero_()
    slots = torch.arange(rows, device="cuda", dtype=torch.int32) % 3
    layer = SimpleNamespace(
        weight=weight,
        es_pre_delta_bank=None,
        es_post_delta_bank=None,
        es_rank1_down_bank=down_bank,
        es_rank1_up_bank=up_bank,
    )
    method = UnquantizedLinearMethod()
    output = torch.empty((rows, n), device="cuda", dtype=torch.bfloat16)

    with forward_context(ForwardContext(attn_backend=None, es_candidate_slots=slots)):
        method.apply_into(layer, x, output, bias)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply_into(layer, x, output, bias)

    assert captured.data_ptr() == output.data_ptr()
    x.copy_(torch.randn_like(x))
    slots.copy_((slots + 1) % 3)
    down_bank[1:].uniform_(-0.02, 0.02)
    up_bank[1:].uniform_(-0.15, 0.15)
    graph.replay()
    torch.cuda.synchronize()

    base_output = triton_bf16_linear(x, weight, bias)
    expected = _dense_reference(x, base_output, down_bank, up_bank, slots)
    torch.testing.assert_close(output, expected, rtol=8e-3, atol=2e-2)


def test_block_fp8_base_gemm_plus_rank1_uses_original_bf16_activation():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("block FP8 Triton GEMM requires Hopper or newer")

    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        triton_w8a8_block_fp8_linear,
    )
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    torch.manual_seed(20260901)
    rows, k, n = 9, 256, 256
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    weight_scale = torch.full((2, 2), 0.01, device="cuda", dtype=torch.float32)
    weight = (
        torch.randn((n, k), device="cuda", dtype=torch.float32)
        .div(weight_scale[0, 0])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    base_output = triton_w8a8_block_fp8_linear(
        input=x,
        weight=weight,
        block_size=[128, 128],
        weight_scale=weight_scale,
    )
    original_base = base_output.clone()
    down_bank = torch.randn((3, k), device="cuda", dtype=torch.float32) * 0.01
    up_bank = torch.randn((3, n), device="cuda", dtype=torch.float32) * 0.1
    down_bank[0].zero_()
    up_bank[0].zero_()
    slots = torch.arange(rows, device="cuda", dtype=torch.int32) % 3

    layer = SimpleNamespace(
        weight=weight,
        weight_scale_inv=weight_scale,
        es_rank1_down_bank=down_bank,
        es_rank1_up_bank=up_bank,
        es_post_delta_bank=None,
    )
    method = SimpleNamespace(weight_block_size=[128, 128])
    with forward_context(ForwardContext(attn_backend=None, es_candidate_slots=slots)):
        actual = Fp8LinearMethod.apply(method, layer, x)
    expected = _dense_reference(x, original_base, down_bank, up_bank, slots)

    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-2)
    assert torch.equal(actual[slots == 0], original_base[slots == 0])


@pytest.mark.parametrize("input_route_major", [False, True])
@pytest.mark.parametrize("apply_router_weight", [False, True])
def test_moe_rank1_mixed_slots_experts_and_router_weight_ordering(
    input_route_major, apply_router_weight
):
    from sglang.srt.diag_es.moe_ops import apply_moe_rank1_residual

    torch.manual_seed(20260902)
    tokens, topk, experts, slots_count, k, n = 5, 2, 3, 3, 263, 389
    topk_ids = torch.tensor(
        [[0, 1], [2, -1], [1, 2], [0, 2], [2, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(
        torch.randn((tokens, topk), device="cuda", dtype=torch.float32), dim=-1
    ).contiguous()
    slots = torch.tensor([0, 1, 2, 1, 0], device="cuda", dtype=torch.int32)
    rows = tokens * topk if input_route_major else tokens
    activations = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    base_output = torch.randn((tokens, topk, n), device="cuda", dtype=torch.bfloat16)
    original_base = base_output.clone()
    down_bank = (
        torch.randn((experts, slots_count, k), device="cuda", dtype=torch.float32)
        * 0.01
    )
    up_bank = (
        torch.randn((experts, slots_count, n), device="cuda", dtype=torch.float32) * 0.1
    )
    down_bank[:, 0].zero_()
    up_bank[:, 0].zero_()

    returned = apply_moe_rank1_residual(
        activations,
        base_output,
        topk_ids,
        topk_weights,
        slots,
        down_bank,
        up_bank,
        input_route_major=input_route_major,
        apply_router_weight=apply_router_weight,
    )
    expected = _moe_reference(
        activations,
        original_base,
        topk_ids,
        topk_weights,
        slots,
        down_bank,
        up_bank,
        input_route_major=input_route_major,
        apply_router_weight=apply_router_weight,
    )

    assert returned.data_ptr() == base_output.data_ptr()
    torch.testing.assert_close(base_output, expected, rtol=8e-3, atol=2e-2)
    assert torch.equal(base_output[slots == 0], original_base[slots == 0])
    invalid_route = topk_ids < 0
    assert torch.equal(base_output[invalid_route], original_base[invalid_route])


def test_moe_rank1_cuda_graph_replay_observes_live_routes_slots_and_factors():
    from sglang.srt.diag_es.moe_ops import apply_moe_rank1_residual

    torch.manual_seed(20260903)
    tokens, topk, experts, slots_count, k, n = 7, 2, 3, 3, 256, 256
    activations = torch.randn((tokens * topk, k), device="cuda", dtype=torch.bfloat16)
    base_output = torch.randn((tokens, topk, n), device="cuda", dtype=torch.bfloat16)
    topk_ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32) % experts
    ).view(tokens, topk)
    topk_weights = torch.softmax(
        torch.randn((tokens, topk), device="cuda", dtype=torch.float32), dim=-1
    ).contiguous()
    slots = torch.arange(tokens, device="cuda", dtype=torch.int32) % slots_count
    down_bank = (
        torch.randn((experts, slots_count, k), device="cuda", dtype=torch.float32)
        * 0.01
    )
    up_bank = (
        torch.randn((experts, slots_count, n), device="cuda", dtype=torch.float32) * 0.1
    )

    apply_moe_rank1_residual(
        activations,
        base_output,
        topk_ids,
        topk_weights,
        slots,
        down_bank,
        up_bank,
        input_route_major=True,
        apply_router_weight=True,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_moe_rank1_residual(
            activations,
            base_output,
            topk_ids,
            topk_weights,
            slots,
            down_bank,
            up_bank,
            input_route_major=True,
            apply_router_weight=True,
        )

    activations.copy_(torch.randn_like(activations))
    base_output.copy_(torch.randn_like(base_output))
    original_base = base_output.clone()
    topk_ids.copy_((topk_ids + 1) % experts)
    topk_weights.copy_(torch.softmax(torch.randn_like(topk_weights), dim=-1))
    slots.copy_((slots + 1) % slots_count)
    down_bank.uniform_(-0.02, 0.02)
    up_bank.uniform_(-0.15, 0.15)
    graph.replay()
    torch.cuda.synchronize()

    expected = _moe_reference(
        activations,
        original_base,
        topk_ids,
        topk_weights,
        slots,
        down_bank,
        up_bank,
        input_route_major=True,
        apply_router_weight=True,
    )
    torch.testing.assert_close(base_output, expected, rtol=8e-3, atol=2e-2)


@pytest.mark.parametrize(
    ("input_sorted", "output_sorted", "apply_router_weight"),
    [(False, True, False), (True, False, True)],
)
def test_moe_rank1_preserves_tma_sorted_intermediate_layout(
    input_sorted, output_sorted, apply_router_weight
):
    from sglang.srt.diag_es.moe_ops import apply_moe_rank1_residual

    torch.manual_seed(20260904 + int(input_sorted))
    tokens, topk, experts, slots_count, k, n = 4, 2, 3, 3, 263, 389
    routes = tokens * topk
    sorted_rows = 12
    sorted_token_ids = torch.tensor(
        [5, 0, 7, routes, 3, 1, 2, 4, 6, routes, routes, routes],
        device="cuda",
        dtype=torch.int32,
    )
    num_tokens_post_padded = torch.tensor([10], device="cuda", dtype=torch.int32)
    topk_ids = torch.tensor(
        [[0, 1], [2, -1], [1, 2], [0, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(
        torch.randn((tokens, topk), device="cuda", dtype=torch.float32), dim=-1
    ).contiguous()
    slots = torch.tensor([0, 1, 2, 1], device="cuda", dtype=torch.int32)
    activation_rows = sorted_rows if input_sorted else tokens
    output_rows = sorted_rows if output_sorted else routes
    activations = torch.randn((activation_rows, k), device="cuda", dtype=torch.bfloat16)
    base_output = torch.randn((output_rows, n), device="cuda", dtype=torch.bfloat16)
    original_base = base_output.clone()
    down_bank = (
        torch.randn((experts, slots_count, k), device="cuda", dtype=torch.float32)
        * 0.01
    )
    up_bank = (
        torch.randn((experts, slots_count, n), device="cuda", dtype=torch.float32) * 0.1
    )
    down_bank[:, 0].zero_()
    up_bank[:, 0].zero_()

    apply_moe_rank1_residual(
        activations,
        base_output,
        topk_ids,
        topk_weights,
        slots,
        down_bank,
        up_bank,
        input_route_major=input_sorted,
        apply_router_weight=apply_router_weight,
        sorted_token_ids=sorted_token_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        input_sorted=input_sorted,
        output_sorted=output_sorted,
    )
    expected = _moe_sorted_reference(
        activations,
        original_base,
        sorted_token_ids,
        num_tokens_post_padded,
        topk_ids,
        topk_weights,
        slots,
        down_bank,
        up_bank,
        input_sorted=input_sorted,
        output_sorted=output_sorted,
        apply_router_weight=apply_router_weight,
    )

    torch.testing.assert_close(base_output, expected, rtol=8e-3, atol=2e-2)
    if output_sorted:
        # Padding inside the used prefix and allocator tail rows are untouched.
        assert torch.equal(base_output[[3, 9, 10, 11]], original_base[[3, 9, 10, 11]])
    else:
        # Route 3 has expert=-1 and must retain the native GEMM row.
        assert torch.equal(base_output[3], original_base[3])
