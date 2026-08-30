from __future__ import annotations

import torch
import triton
import triton.language as tl
from sglang.srt.model_executor.forward_context import get_forward_context


@triton.jit
def _apply_dense_delta_kernel(
    x_ptr,
    delta_ptr,
    slot_ptr,
    out_ptr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    slot = tl.load(slot_ptr + row)
    x = tl.load(x_ptr + row * width + cols, mask=mask)
    delta = tl.load(delta_ptr + slot * width + cols, mask=mask)
    x_fp32 = x.to(tl.float32)
    # Keep the residual multiply/add on CUDA FP32 cores.  The activation is
    # rounded only once, when the completed FP32 result is stored as BF16.
    steered_fp32 = tl.fma(x_fp32, delta, x_fp32)
    tl.store(
        out_ptr + row * width + cols,
        steered_fp32.to(x.dtype),
        mask=mask,
    )


def _dense_delta_launch_config(rows: int, width: int) -> tuple[int, int | None]:
    """Return the offline-tuned B300 launch geometry for target dense sites."""

    if width not in (2048, 4096) or rows < 512:
        return 256, None
    if rows < 1024:
        return (256, None) if width == 2048 else (512, 4)
    if width == 2048:
        if rows < 2048:
            return 512, 4
        if rows < 4096:
            return 2048, 4
        return 2048, 8
    if rows < 2048:
        return 2048, 4
    if rows < 4096:
        return 2048, 8
    return 4096, 8


def _launch_dense_delta(
    x: torch.Tensor,
    delta_bank: torch.Tensor,
    candidate_slots: torch.Tensor,
    out: torch.Tensor,
) -> None:
    width = x.shape[1]
    block, num_warps = _dense_delta_launch_config(x.shape[0], width)
    grid = (x.shape[0], triton.cdiv(width, block))
    if num_warps is None:
        _apply_dense_delta_kernel[grid](
            x,
            delta_bank,
            candidate_slots,
            out,
            width=width,
            BLOCK=block,
        )
    else:
        _apply_dense_delta_kernel[grid](
            x,
            delta_bank,
            candidate_slots,
            out,
            width=width,
            BLOCK=block,
            num_warps=num_warps,
        )


def apply_dense_delta_out(
    x: torch.Tensor,
    delta_bank: torch.Tensor,
    candidate_slots: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _launch_dense_delta(x, delta_bank, candidate_slots, out)
    return out


def apply_dense_delta(
    x: torch.Tensor, delta_bank: torch.Tensor, candidate_slots: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    _launch_dense_delta(x, delta_bank, candidate_slots, out)
    return out


def maybe_apply_diag_es_pre(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    delta_bank = layer.es_pre_delta_bank
    if delta_bank is None:
        return x
    candidate_slots = get_forward_context().es_candidate_slots
    if candidate_slots is None:
        raise RuntimeError(
            "diagonal-ES pre perturbation is active but candidate slots are missing"
        )
    return apply_dense_delta(x, delta_bank, candidate_slots)


def get_diag_es_post_inputs(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_slots = get_forward_context().es_candidate_slots
    if candidate_slots is None:
        raise RuntimeError(
            "diagonal-ES post perturbation is active but candidate slots are missing"
        )
    return layer.es_post_delta_bank, candidate_slots


@triton.jit
def _apply_dense_rank1_residual_kernel(
    x_ptr,
    base_output_ptr,
    down_bank_ptr,
    up_bank_ptr,
    slot_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Add one slot-selected rank-1 residual to one dense output row."""

    row = tl.program_id(0)
    slot = tl.load(slot_ptr + row).to(tl.int64)

    rank1_scalar = 0.0
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        activation = tl.load(
            x_ptr + row * K + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        down = tl.load(
            down_bank_ptr + slot * K + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        rank1_scalar += tl.sum(activation * down, axis=0)

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        base_output = tl.load(
            base_output_ptr + row * N + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            up_bank_ptr + slot * N + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        output = tl.fma(rank1_scalar, up, base_output)
        tl.store(out_ptr + row * N + n_offsets, output, mask=n_mask)


def rank1_launch_config(row_count: int) -> tuple[int, int, int]:
    """Return the B300-tuned Qwen rank-1 launch geometry.

    The same threshold holds for all four supported Qwen dense and routed-MoE
    shapes. MoE callers pass routed rows, not activation-token rows.
    """

    if row_count <= 1024:
        return 512, 512, 8
    return 256, 256, 4


def apply_dense_rank1_out(
    x: torch.Tensor,
    base_output: torch.Tensor,
    down_bank: torch.Tensor,
    up_bank: torch.Tensor,
    candidate_slots: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Compute ``base_output + (x @ down) * up`` in FP32, then store BF16."""

    x_2d = x.view(-1, x.shape[-1])
    base_output_2d = base_output.view(x_2d.shape[0], -1)
    out_2d = out.view_as(base_output_2d)
    block_k, block_n, num_warps = rank1_launch_config(x_2d.shape[0])
    _apply_dense_rank1_residual_kernel[(x_2d.shape[0],)](
        x_2d,
        base_output_2d,
        down_bank,
        up_bank,
        candidate_slots,
        out_2d,
        K=x_2d.shape[1],
        N=base_output_2d.shape[1],
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def apply_dense_rank1_inplace(
    x: torch.Tensor,
    base_output: torch.Tensor,
    down_bank: torch.Tensor,
    up_bank: torch.Tensor,
    candidate_slots: torch.Tensor,
) -> torch.Tensor:
    """Apply a dense rank-1 residual in place on ``base_output``."""

    return apply_dense_rank1_out(
        x,
        base_output,
        down_bank,
        up_bank,
        candidate_slots,
        base_output,
    )


def maybe_apply_rank1_es(
    layer: torch.nn.Module,
    x: torch.Tensor,
    base_output: torch.Tensor,
) -> torch.Tensor:
    """Apply the configured dense rank-1 ES residual after the base GEMM."""

    down_bank = layer.es_rank1_down_bank
    if down_bank is None:
        return base_output
    candidate_slots = get_forward_context().es_candidate_slots
    if candidate_slots is None:
        raise RuntimeError(
            "rank-1 ES perturbation is active but candidate slots are missing"
        )
    return apply_dense_rank1_inplace(
        x,
        base_output,
        down_bank,
        layer.es_rank1_up_bank,
        candidate_slots,
    )
