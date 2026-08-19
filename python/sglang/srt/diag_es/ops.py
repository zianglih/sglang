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


def _validate_dense_delta_inputs(
    x: torch.Tensor,
    delta_bank: torch.Tensor,
    candidate_slots: torch.Tensor,
    out: torch.Tensor | None = None,
) -> None:
    assert x.ndim == 2 and x.is_contiguous()
    assert delta_bank.ndim == 2 and delta_bank.is_contiguous()
    assert candidate_slots.ndim == 1 and candidate_slots.is_contiguous()
    assert x.dtype == torch.bfloat16 and delta_bank.dtype == torch.float32
    assert candidate_slots.dtype == torch.int32
    assert x.device == delta_bank.device == candidate_slots.device
    assert x.shape[0] > 0 and x.shape[1] > 0 and delta_bank.shape[0] > 0
    assert candidate_slots.shape[0] == x.shape[0]
    assert delta_bank.shape[1] == x.shape[1]
    if out is not None:
        assert out.ndim == 2 and out.is_contiguous()
        assert out.dtype == x.dtype and out.shape == x.shape
        assert out.device == x.device


def apply_dense_delta_out(
    x: torch.Tensor,
    delta_bank: torch.Tensor,
    candidate_slots: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _validate_dense_delta_inputs(x, delta_bank, candidate_slots, out)
    _launch_dense_delta(x, delta_bank, candidate_slots, out)
    return out


def apply_dense_delta(
    x: torch.Tensor, delta_bank: torch.Tensor, candidate_slots: torch.Tensor
) -> torch.Tensor:
    _validate_dense_delta_inputs(x, delta_bank, candidate_slots)
    out = torch.empty_like(x)
    _launch_dense_delta(x, delta_bank, candidate_slots, out)
    return out


def get_dense_delta_inputs(
    layer: torch.nn.Module, position: str
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Return the live slot tensor and fixed-address bank for one dense site."""

    if position not in ("pre", "post"):
        raise ValueError("dense diagonal-ES position must be pre or post")
    site_id = getattr(layer, f"es_{position}_site_id", None)
    delta_bank = getattr(layer, f"es_{position}_delta_bank", None)
    assert (site_id is None) == (delta_bank is None)
    if delta_bank is None:
        return None, None
    slots = get_forward_context().es_candidate_slots
    assert slots is not None
    return delta_bank, slots


def maybe_apply_diag_es_pre(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    delta_bank, slots = get_dense_delta_inputs(layer, "pre")
    if delta_bank is None:
        return x
    return apply_dense_delta(x, delta_bank, slots)


def get_diag_es_post_inputs(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    return get_dense_delta_inputs(layer, "post")
