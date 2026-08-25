"""Qwen3-30B-A3B target-shape tests for the Triton BF16 linear."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.gemm.triton_bf16_gemm import (  # noqa: E402
    _triton_bf16_linear_kernel,
    triton_bf16_linear,
    triton_bf16_linear_out,
)

TARGET_SHAPES = [
    (1, 5120, 2048),
    (32, 2048, 4096),
    (257, 128, 2048),
    (1, 151936, 2048),
]

JOYAI_MTP_IDENTITY_CASES = [
    # M spans single-request decode through a full 64-session draft batch.
    # (N, K) covers every BF16 dense linear in JoyAI's NextN decoder.
    (1, 2112, 2048),
    (2, 6144, 1536),
    (3, 2048, 4096),
    (17, 14336, 2048),
    (64, 2048, 7168),
]


def test_triton_bf16_linear_reuses_kernel_across_dynamic_batch_sizes():
    assert _triton_bf16_linear_kernel.keys == ["N", "K", "APPLY_POST_DELTA"]
    assert "M" in _triton_bf16_linear_kernel.fn.do_not_specialize


def test_triton_bf16_linear_dynamic_batch_reuses_binary_in_cuda_graph():
    torch.manual_seed(20260819)
    n, k = 128, 2048
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    # Compile/autotune only M=1 before capture. Capturing a previously unseen
    # M=17 succeeds only when M is not a specialization or autotune key.
    x1 = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
    out1 = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    triton_bf16_linear_out(x1, weight, out1)
    torch.cuda.synchronize()

    x17 = torch.randn((17, k), dtype=torch.bfloat16, device="cuda")
    out17 = torch.empty((17, n), dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        triton_bf16_linear_out(x17, weight, out17)
    graph.replay()
    torch.cuda.synchronize()

    ref = (x17.float() @ weight.float().T).bfloat16()
    torch.testing.assert_close(out17, ref, rtol=2e-2, atol=2.5)


def test_triton_bf16_linear_post_delta_dynamic_batch_cuda_graph():
    torch.manual_seed(20260820)
    n, k = 128, 2048
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    delta_bank = torch.zeros((3, n), dtype=torch.float32, device="cuda")
    delta_bank[1:].uniform_(-0.02, 0.02)

    # Warm the post-delta specialization before capture; M remains dynamic.
    x1 = torch.randn((1, k), dtype=torch.bfloat16, device="cuda")
    out1 = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    slots1 = torch.ones(1, dtype=torch.int32, device="cuda")
    triton_bf16_linear_out(
        x1,
        weight,
        out1,
        post_delta_bank=delta_bank,
        candidate_slots=slots1,
    )
    torch.cuda.synchronize()

    x17 = torch.randn((17, k), dtype=torch.bfloat16, device="cuda")
    out17 = torch.empty((17, n), dtype=torch.bfloat16, device="cuda")
    slots17 = torch.arange(17, dtype=torch.int32, device="cuda") % 3
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        triton_bf16_linear_out(
            x17,
            weight,
            out17,
            post_delta_bank=delta_bank,
            candidate_slots=slots17,
        )

    x17.copy_(torch.randn_like(x17))
    slots17.copy_((torch.arange(17, dtype=torch.int32, device="cuda") + 1) % 3)
    delta_bank[1:].uniform_(-0.03, 0.03)
    graph.replay()
    torch.cuda.synchronize()

    affine = x17.float() @ weight.float().T
    ref = torch.addcmul(
        affine,
        affine,
        delta_bank[slots17.long()],
    ).bfloat16()
    torch.testing.assert_close(out17, ref, rtol=2e-2, atol=2.5)


def test_triton_bf16_linear_post_delta_bias_order_and_identity():
    m = n = k = 64
    x = torch.zeros((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.linspace(-2, 2, n, dtype=torch.bfloat16, device="cuda")
    delta_bank = torch.zeros((3, n), dtype=torch.float32, device="cuda")
    delta_bank[1].fill_(0.125)
    delta_bank[2].fill_(-0.25)
    slots = torch.arange(m, dtype=torch.int32, device="cuda") % 3

    actual = triton_bf16_linear(
        x,
        weight,
        bias,
        post_delta_bank=delta_bank,
        candidate_slots=slots,
    )
    output = torch.empty_like(actual)
    returned = triton_bf16_linear_out(
        x,
        weight,
        output,
        bias,
        post_delta_bank=delta_bank,
        candidate_slots=slots,
    )
    affine = bias.float().expand(m, n)
    expected = torch.addcmul(
        affine,
        affine,
        delta_bank[slots.long()],
    ).bfloat16()

    assert returned.data_ptr() == output.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(output, expected)

    identity_slots = torch.zeros(m, dtype=torch.int32, device="cuda")
    native = triton_bf16_linear(x, weight, bias)
    identity = triton_bf16_linear(
        x,
        weight,
        bias,
        post_delta_bank=delta_bank,
        candidate_slots=identity_slots,
    )
    assert torch.equal(identity.view(torch.int16), native.view(torch.int16))


@pytest.mark.parametrize("m,n,k", JOYAI_MTP_IDENTITY_CASES)
def test_triton_bf16_linear_zero_post_delta_is_bitwise_identity(m, n, k):
    torch.manual_seed(20260825 + m + n + k)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    zero_delta_bank = torch.zeros((5, n), dtype=torch.float32, device="cuda")
    slots = torch.arange(m, dtype=torch.int32, device="cuda") % 5

    native = triton_bf16_linear(x, weight)
    identity = triton_bf16_linear(
        x,
        weight,
        post_delta_bank=zero_delta_bank,
        candidate_slots=slots,
    )

    assert torch.equal(identity.view(torch.int16), native.view(torch.int16))


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("m,n,k", TARGET_SHAPES)
def test_triton_bf16_linear_target_shapes(m, n, k, has_bias):
    torch.manual_seed(m + n + k)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda") if has_bias else None

    out = triton_bf16_linear(x, weight, bias)
    out_buffer = torch.empty_like(out)
    triton_bf16_linear_out(x, weight, out_buffer, bias)

    ref = x.float() @ weight.float().T
    if bias is not None:
        ref += bias.float()
    ref = ref.bfloat16()

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2.5)
    torch.testing.assert_close(out_buffer, ref, rtol=2e-2, atol=2.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
