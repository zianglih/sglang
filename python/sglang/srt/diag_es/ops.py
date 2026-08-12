from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.diag_es.manager import get_diag_es_manager
from sglang.srt.model_executor.forward_context import get_forward_context


@triton.jit
def _apply_dense_gate_kernel(
    x_ptr,
    gate_ptr,
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
    gate = tl.load(gate_ptr + slot * width + cols, mask=mask)
    tl.store(out_ptr + row * width + cols, (x * gate).to(x.dtype), mask=mask)


def apply_dense_gate(
    x: torch.Tensor, gate_bank: torch.Tensor, candidate_slots: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous()
    assert gate_bank.ndim == 2 and gate_bank.is_contiguous()
    assert candidate_slots.ndim == 1 and candidate_slots.is_contiguous()
    assert x.dtype == torch.bfloat16 and gate_bank.dtype == torch.bfloat16
    assert candidate_slots.dtype == torch.int32
    assert x.device == gate_bank.device == candidate_slots.device
    assert candidate_slots.shape[0] == x.shape[0]
    assert gate_bank.shape[1] == x.shape[1]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    width = x.shape[1]
    block = 256
    _apply_dense_gate_kernel[(x.shape[0], triton.cdiv(width, block))](
        x,
        gate_bank,
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
    gate_bank = get_diag_es_manager().get_dense_gate_bank(site_id)
    return apply_dense_gate(x, gate_bank, slots)
