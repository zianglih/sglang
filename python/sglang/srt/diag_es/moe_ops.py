from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.diag_es.ops import rank1_launch_config

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
    fc1_rank1_down: Optional[torch.Tensor] = None
    fc1_rank1_up: Optional[torch.Tensor] = None
    fc2_rank1_down: Optional[torch.Tensor] = None
    fc2_rank1_up: Optional[torch.Tensor] = None


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


@triton.jit
def _apply_moe_rank1_residual_kernel(
    x_ptr,
    base_output_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    token_slots_ptr,
    down_bank_ptr,
    up_bank_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
    INPUT_ROUTE_MAJOR: tl.constexpr,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Add one expert/slot-selected rank-1 residual to one routed row."""

    route = tl.program_id(0)
    token = route // ROUTER_TOPK
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    if (expert < 0) | (expert >= NUM_EXPERTS):
        return

    slot = tl.load(token_slots_ptr + token).to(tl.int64)
    x_row = route if INPUT_ROUTE_MAJOR else token

    rank1_scalar = 0.0
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        activation = tl.load(
            x_ptr + x_row * K + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        down = tl.load(
            down_bank_ptr + (expert * NUM_SLOTS + slot) * K + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        rank1_scalar += tl.sum(activation * down, axis=0)

    if APPLY_ROUTER_WEIGHT:
        route_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        rank1_scalar *= route_weight

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        base_output = tl.load(
            base_output_ptr + route * N + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            up_bank_ptr + (expert * NUM_SLOTS + slot) * N + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        output = tl.fma(rank1_scalar, up, base_output)
        tl.store(out_ptr + route * N + n_offsets, output, mask=n_mask)


@triton.jit
def _apply_moe_sorted_rank1_residual_kernel(
    x_ptr,
    base_output_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    token_slots_ptr,
    down_bank_ptr,
    up_bank_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
    INPUT_SORTED: tl.constexpr,
    OUTPUT_SORTED: tl.constexpr,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply one rank-1 residual at one expert-sorted routing position."""

    sorted_position = tl.program_id(0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if sorted_position >= num_tokens_post_padded:
        return

    route = tl.load(sorted_token_ids_ptr + sorted_position).to(tl.int64)
    if (route < 0) | (route >= NUM_ROUTES):
        return

    token = route // ROUTER_TOPK
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    if (expert < 0) | (expert >= NUM_EXPERTS):
        return

    slot = tl.load(token_slots_ptr + token).to(tl.int64)
    x_row = sorted_position if INPUT_SORTED else token
    output_row = sorted_position if OUTPUT_SORTED else route

    rank1_scalar = 0.0
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        activation = tl.load(
            x_ptr + x_row * K + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        down = tl.load(
            down_bank_ptr + (expert * NUM_SLOTS + slot) * K + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        rank1_scalar += tl.sum(activation * down, axis=0)

    if APPLY_ROUTER_WEIGHT:
        route_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        rank1_scalar *= route_weight

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        base_output = tl.load(
            base_output_ptr + output_row * N + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            up_bank_ptr + (expert * NUM_SLOTS + slot) * N + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        output = tl.fma(rank1_scalar, up, base_output)
        tl.store(out_ptr + output_row * N + n_offsets, output, mask=n_mask)


def apply_moe_rank1_residual(
    activations: torch.Tensor,
    base_output: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_slots: torch.Tensor,
    down_bank: torch.Tensor,
    up_bank: torch.Tensor,
    *,
    input_route_major: bool,
    apply_router_weight: bool,
    sorted_token_ids: Optional[torch.Tensor] = None,
    num_tokens_post_padded: Optional[torch.Tensor] = None,
    input_sorted: bool = False,
    output_sorted: bool = False,
) -> torch.Tensor:
    """Apply a routed rank-1 residual in place without changing the base GEMM."""

    activations_2d = activations.view(-1, activations.shape[-1])
    base_output_2d = base_output.view(-1, base_output.shape[-1])
    block_k, block_n, num_warps = rank1_launch_config(topk_ids.numel())
    if sorted_token_ids is not None:
        _apply_moe_sorted_rank1_residual_kernel[(sorted_token_ids.numel(),)](
            activations_2d,
            base_output_2d,
            sorted_token_ids,
            num_tokens_post_padded,
            topk_ids,
            topk_weights,
            token_slots,
            down_bank,
            up_bank,
            base_output_2d,
            K=activations_2d.shape[1],
            N=base_output_2d.shape[1],
            NUM_ROUTES=topk_ids.numel(),
            NUM_EXPERTS=down_bank.shape[0],
            NUM_SLOTS=down_bank.shape[1],
            ROUTER_TOPK=topk_ids.shape[1],
            INPUT_SORTED=input_sorted,
            OUTPUT_SORTED=output_sorted,
            APPLY_ROUTER_WEIGHT=apply_router_weight,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return base_output

    _apply_moe_rank1_residual_kernel[(topk_ids.numel(),)](
        activations_2d,
        base_output_2d,
        topk_ids,
        topk_weights,
        token_slots,
        down_bank,
        up_bank,
        base_output_2d,
        K=activations_2d.shape[1],
        N=base_output_2d.shape[1],
        NUM_EXPERTS=down_bank.shape[0],
        NUM_SLOTS=down_bank.shape[1],
        ROUTER_TOPK=topk_ids.shape[1],
        INPUT_ROUTE_MAJOR=input_route_major,
        APPLY_ROUTER_WEIGHT=apply_router_weight,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return base_output
