from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_expert_route_gate_kernel(
    x_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    gate_ptr,
    out_ptr,
    width: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_ge,
    stride_gs,
    stride_gk,
    stride_om,
    stride_ok,
    ROUTER_TOPK: tl.constexpr,
    INPUT_ROUTE_MAJOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    route = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token = route // ROUTER_TOPK
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    slot = tl.load(token_slots_ptr + token).to(tl.int64)
    valid = (cols < width) & (expert >= 0)

    x_row = route if INPUT_ROUTE_MAJOR else token
    x = tl.load(
        x_ptr + x_row * stride_xm + cols * stride_xk,
        mask=valid,
        other=0.0,
    )
    gate = tl.load(
        gate_ptr + expert * stride_ge + slot * stride_gs + cols * stride_gk,
        mask=valid,
        other=0.0,
    )
    out_dtype = out_ptr.dtype.element_ty
    gated = (x * gate).to(out_dtype)
    tl.store(
        out_ptr + route * stride_om + cols * stride_ok,
        gated,
        mask=cols < width,
    )


def _launch_expert_route_gate(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    gate_bank: torch.Tensor,
    out: torch.Tensor,
    *,
    input_route_major: bool,
) -> None:
    width = x.shape[1]
    block_size = 256
    grid = (topk_ids.numel(), triton.cdiv(width, block_size))
    _apply_expert_route_gate_kernel[grid](
        x,
        topk_ids,
        token_slots,
        gate_bank,
        out,
        width=width,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_ge=gate_bank.stride(0),
        stride_gs=gate_bank.stride(1),
        stride_gk=gate_bank.stride(2),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        ROUTER_TOPK=topk_ids.shape[1],
        INPUT_ROUTE_MAJOR=input_route_major,
        BLOCK_SIZE=block_size,
    )


def materialize_moe_fc1_input(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    gate_bank: torch.Tensor,
) -> torch.Tensor:
    """Expand token-major hidden states into gated route-major FC1 input."""

    out = torch.empty(
        (topk_ids.numel(), hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _launch_expert_route_gate(
        hidden_states,
        topk_ids,
        token_slots,
        gate_bank,
        out,
        input_route_major=False,
    )
    return out


def apply_moe_fc2_gate_inplace(
    activations: torch.Tensor,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    gate_bank: torch.Tensor,
) -> torch.Tensor:
    """Gate the dead post-activation route buffer immediately before FC2."""

    _launch_expert_route_gate(
        activations,
        topk_ids,
        token_slots,
        gate_bank,
        activations,
        input_route_major=True,
    )
    return activations
