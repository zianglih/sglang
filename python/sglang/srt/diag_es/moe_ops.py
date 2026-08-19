from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig


@dataclass(frozen=True, slots=True)
class MoeDeltaBanks:
    """Fixed-address MoE delta views for one layer and forward."""

    token_slots: Optional[torch.Tensor] = None
    fc1_pre: Optional[torch.Tensor] = None
    fc1_post: Optional[torch.Tensor] = None
    fc2_pre: Optional[torch.Tensor] = None
    fc2_post: Optional[torch.Tensor] = None


EMPTY_MOE_DELTA_BANKS = MoeDeltaBanks()


def get_moe_delta_banks(moe_runner_config: MoeRunnerConfig) -> MoeDeltaBanks:
    """Resolve the live token slots and resident banks for one MoE layer."""

    fixed_banks = moe_runner_config.diag_es_delta_banks
    if fixed_banks is None:
        return EMPTY_MOE_DELTA_BANKS

    from sglang.srt.model_executor.forward_context import get_forward_context

    token_slots = get_forward_context().es_candidate_slots
    return replace(fixed_banks, token_slots=token_slots)


@triton.jit
def _apply_expert_route_pre_delta_kernel(
    x_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    delta_ptr,
    out_ptr,
    width: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
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
        x_ptr + x_row * width + cols,
        mask=valid,
        other=0.0,
    )
    delta = tl.load(
        delta_ptr + (expert * NUM_SLOTS + slot) * width + cols,
        mask=valid,
        other=0.0,
    )
    x_fp32 = x.to(tl.float32)
    steered = tl.fma(x_fp32, delta, x_fp32).to(tl.bfloat16)
    tl.store(
        out_ptr + route * width + cols,
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
    _apply_expert_route_pre_delta_kernel[grid](
        x,
        topk_ids,
        token_slots,
        delta_bank,
        out,
        width=width,
        NUM_SLOTS=delta_bank.shape[1],
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
    """Expand token-major states into pre-steered route-major FC1 input."""

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
    """Steer the dead post-activation route buffer immediately before FC2."""

    _launch_expert_route_pre_delta(
        activations,
        topk_ids,
        token_slots,
        delta_bank,
        activations,
        input_route_major=True,
    )
    return activations
