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
