from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_expert_route_pre_delta_kernel(
    x_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    delta_ptr,
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
    delta = tl.load(
        delta_ptr + expert * stride_ge + slot * stride_gs + cols * stride_gk,
        mask=valid,
        other=0.0,
    )
    out_dtype = out_ptr.dtype.element_ty
    x_fp32 = x.to(tl.float32)
    steered = tl.fma(x_fp32, delta, x_fp32).to(out_dtype)
    tl.store(
        out_ptr + route * stride_om + cols * stride_ok,
        steered,
        mask=cols < width,
    )


def _launch_expert_route_pre_delta(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    delta_bank: torch.Tensor,
    out: torch.Tensor,
    *,
    input_route_major: bool,
) -> None:
    width = x.shape[1]
    block_size = 256
    grid = (topk_ids.numel(), triton.cdiv(width, block_size))
    assert x.ndim == 2 and x.is_contiguous()
    assert out.ndim == 2 and out.is_contiguous()
    assert topk_ids.ndim == 2 and topk_ids.is_contiguous()
    assert token_slots.ndim == 1 and token_slots.is_contiguous()
    assert delta_bank.ndim == 3 and delta_bank.is_contiguous()
    assert x.dtype == out.dtype == torch.bfloat16
    assert delta_bank.dtype == torch.float32
    assert topk_ids.dtype == token_slots.dtype == torch.int32
    assert x.device == out.device == delta_bank.device
    assert x.device == topk_ids.device == token_slots.device
    assert token_slots.shape[0] == topk_ids.shape[0]
    assert delta_bank.shape[2] == width
    _apply_expert_route_pre_delta_kernel[grid](
        x,
        topk_ids,
        token_slots,
        delta_bank,
        out,
        width=width,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_ge=delta_bank.stride(0),
        stride_gs=delta_bank.stride(1),
        stride_gk=delta_bank.stride(2),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        ROUTER_TOPK=topk_ids.shape[1],
        INPUT_ROUTE_MAJOR=input_route_major,
        BLOCK_SIZE=block_size,
    )


def materialize_moe_fc1_pre_input(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    delta_bank: torch.Tensor,
) -> torch.Tensor:
    """Expand token-major states into pre-gated route-major FC1 input."""

    out = torch.empty(
        (topk_ids.numel(), hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _launch_expert_route_pre_delta(
        hidden_states,
        topk_ids,
        token_slots,
        delta_bank,
        out,
        input_route_major=False,
    )
    return out


def apply_moe_fc2_pre_delta_inplace(
    activations: torch.Tensor,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    delta_bank: torch.Tensor,
) -> torch.Tensor:
    """Pre-gate the dead post-activation route buffer immediately before FC2."""

    _launch_expert_route_pre_delta(
        activations,
        topk_ids,
        token_slots,
        delta_bank,
        activations,
        input_route_major=True,
    )
    return activations
