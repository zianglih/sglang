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


def test_triton_bf16_linear_reuses_kernel_across_dynamic_batch_sizes():
    assert _triton_bf16_linear_kernel.keys == ["N", "K"]
    assert "M" in _triton_bf16_linear_kernel.fn.do_not_specialize


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
