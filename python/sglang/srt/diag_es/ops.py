from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.diag_es.manager import get_diag_es_manager
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


def apply_dense_delta(
    x: torch.Tensor, delta_bank: torch.Tensor, candidate_slots: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous()
    assert delta_bank.ndim == 2 and delta_bank.is_contiguous()
    assert candidate_slots.ndim == 1 and candidate_slots.is_contiguous()
    assert x.dtype == torch.bfloat16 and delta_bank.dtype == torch.float32
    assert candidate_slots.dtype == torch.int32
    assert x.device == delta_bank.device == candidate_slots.device
    assert candidate_slots.shape[0] == x.shape[0]
    assert delta_bank.shape[1] == x.shape[1]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    width = x.shape[1]
    block = 256
    _apply_dense_delta_kernel[(x.shape[0], triton.cdiv(width, block))](
        x,
        delta_bank,
        candidate_slots,
        out,
        width=width,
        BLOCK=block,
    )
    return out


def maybe_apply_diag_es(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    site_id = getattr(layer, "es_site_id", None)
    if site_id is None:
        return x
    slots = get_forward_context().es_candidate_slots
    delta_bank = get_diag_es_manager().get_dense_delta_bank(site_id)
    return apply_dense_delta(x, delta_bank, slots)
