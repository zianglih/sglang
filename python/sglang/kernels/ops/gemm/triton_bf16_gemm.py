"""Target-only Triton BF16 linear kernels used by diagonal-ES training.

The weight follows ``torch.nn.functional.linear`` layout ``[N, K]``.  This
module intentionally contains no backend fallback: the project harness selects
it only for contiguous CUDA BF16 tensors from Qwen3-30B-A3B.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8,
        num_stages=4,
    ),
]


@triton.autotune(configs=_MATMUL_CONFIGS, key=["N", "K", "APPLY_POST_DELTA"])
@triton.jit(do_not_specialize=["M"])
def _triton_bf16_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    post_delta_ptr,
    candidate_slots_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    stride_ds: tl.constexpr,
    stride_dn: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_POST_DELTA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Stored weight is [N, K]; construct a [K, N] tile for tl.dot.
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = k_start * BLOCK_K + offs_k < K
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        weight_t = tl.load(
            weight_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(x, weight_t, acc=accumulator)
        x_ptrs += BLOCK_K * stride_xk
        weight_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    if APPLY_POST_DELTA:
        row_mask = offs_m < M
        slots = tl.load(candidate_slots_ptr + offs_m, mask=row_mask, other=0)
        delta_ptrs = (
            post_delta_ptr + slots[:, None] * stride_ds + offs_n[None, :] * stride_dn
        )
        delta = tl.load(
            delta_ptrs,
            mask=row_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        # The tensor-core accumulator and bias-adjusted affine value are FP32.
        # Apply the zero-centered residual in FP32, then round exactly once at
        # the existing BF16 store below.
        accumulator = tl.fma(accumulator, delta, accumulator)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        accumulator.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_bf16_linear_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    post_delta_bank: Optional[torch.Tensor] = None,
    candidate_slots: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` into caller-owned BF16 storage."""

    x_2d = x.view(-1, x.shape[-1])
    m, k = x_2d.shape
    n = weight.shape[0]
    output_2d = output.view(m, n)
    apply_post_delta = post_delta_bank is not None

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _triton_bf16_linear_kernel[grid](
        x_2d,
        weight,
        bias if bias is not None else x_2d,
        output_2d,
        post_delta_bank if post_delta_bank is not None else x_2d,
        candidate_slots if candidate_slots is not None else x_2d,
        m,
        n,
        k,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        post_delta_bank.stride(0) if post_delta_bank is not None else 0,
        post_delta_bank.stride(1) if post_delta_bank is not None else 0,
        HAS_BIAS=bias is not None,
        APPLY_POST_DELTA=apply_post_delta,
    )
    return output


def triton_bf16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    post_delta_bank: Optional[torch.Tensor] = None,
    candidate_slots: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``torch.nn.functional.linear``-compatible BF16 output."""

    output = torch.empty(
        (*x.shape[:-1], weight.shape[0]),
        device=x.device,
        dtype=x.dtype,
    )
    return triton_bf16_linear_out(
        x,
        weight,
        output,
        bias,
        post_delta_bank=post_delta_bank,
        candidate_slots=candidate_slots,
    )
