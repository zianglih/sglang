from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, unwrap_write_loc
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


logger = logging.getLogger(__name__)

MTP_REPLAY_CHUNK_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class MTPKVReplayBatchStats:
    replayed_rows: int
    transitioned_requests: int
    enqueue_time_ms: float
    request_rows: tuple[int, ...]


def mtp_replay_bytes_per_token(hidden_size: int) -> int:
    """Bytes/token for the candidate-neutral BF16 activation and position."""

    if hidden_size <= 0:
        raise ValueError("MTP replay hidden_size must be positive")
    return (
        hidden_size * torch.tensor([], dtype=torch.bfloat16).element_size()
        + torch.tensor([], dtype=torch.int64).element_size()
    )


def is_target_driven_mtp_forward(forward_mode: ForwardMode) -> bool:
    """Whether an MTP forward has a candidate-neutral pre-attention input."""

    return forward_mode in (ForwardMode.EXTEND, ForwardMode.DRAFT_EXTEND_V2)


def _effective_candidate_key(status: dict) -> tuple[str, ...]:
    try:
        sigma = float(status["sigma"])
        learning_rate = float(status["learning_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "MTP candidate transition status is missing sigma/learning_rate"
        ) from exc
    if sigma == 0.0:
        if learning_rate != 0.0:
            raise RuntimeError(
                "MTP candidate transition saw sigma=0 with nonzero learning rate"
            )
        return ("identity",)
    try:
        return (
            "perturbed",
            str(int(status["theta_version"])),
            str(int(status["perturbation_seed"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "MTP candidate transition status is missing theta/seed identity"
        ) from exc


def mtp_candidate_requires_kv_replay(previous: dict, current: dict) -> bool:
    """Whether two session statuses represent different effective deltas."""

    if previous is None:
        raise RuntimeError("MTP candidate transition is missing prior request status")
    return _effective_candidate_key(previous) != _effective_candidate_key(current)


class DiagESMTPDraftKVReplay:
    """Rebuild JoyAI draft MLA KV whenever an effective candidate changes.

    The cached tensor is the BF16 input to the fused q_a/kv_a projection.  It
    is candidate-neutral only on target-driven prompt/draft-extend forwards;
    draft-decode recursively consumes drafter hidden states and must never
    overwrite this sidecar.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        token_to_kv_pool: MLATokenToKVPool,
        chunk_tokens: int = MTP_REPLAY_CHUNK_TOKENS,
    ) -> None:
        if model.__class__.__name__ != "JoyAILLMFlashForCausalLMNextN":
            raise TypeError(
                "MTP draft-KV replay requires JoyAILLMFlashForCausalLMNextN"
            )
        if getattr(model, "quant_config", None) is not None:
            raise ValueError("MTP draft-KV replay requires an unquantized BF16 draft")
        if not isinstance(token_to_kv_pool, MLATokenToKVPool):
            raise TypeError("MTP draft-KV replay requires a direct MLATokenToKVPool")
        if token_to_kv_pool.dtype != torch.bfloat16:
            raise ValueError("MTP draft-KV replay requires a BF16 draft KV cache")
        if token_to_kv_pool.layer_num != 1:
            raise ValueError("MTP draft-KV replay requires one draft MLA layer")
        if (
            token_to_kv_pool.kv_lora_rank != 512
            or token_to_kv_pool.qk_rope_head_dim != 64
        ):
            raise ValueError("MTP draft-KV replay requires JoyAI MLA dimensions 512+64")
        if chunk_tokens < 1:
            raise ValueError("MTP draft-KV replay chunk_tokens must be positive")

        decoder = model.model.decoder
        attention = decoder.self_attn
        fused_projection = attention.fused_qkv_a_proj_with_mqa
        if (
            fused_projection.es_pre_delta_bank is None
            and fused_projection.es_post_delta_bank is None
        ):
            raise ValueError(
                "MTP draft-KV replay requires fused_qkv_a_proj_with_mqa steering"
            )
        if tuple(fused_projection.weight.shape) != (2112, 2048):
            raise ValueError("MTP draft-KV replay requires JoyAI fused projection")
        if fused_projection.weight.dtype != torch.bfloat16:
            raise ValueError("MTP draft-KV replay requires BF16 fused weights")

        self.fused_projection = fused_projection
        self.kv_a_layernorm = attention.kv_a_layernorm
        self.rotary_emb = attention.rotary_emb
        self.attn_mqa = attention.attn_mqa
        self.q_lora_rank = attention.q_lora_rank
        self.kv_lora_rank = attention.kv_lora_rank
        self.token_to_kv_pool = token_to_kv_pool
        self.hidden_size = 2048
        self.chunk_tokens = chunk_tokens
        self.num_slots = token_to_kv_pool.size + token_to_kv_pool.page_size

        pool_context = (
            torch.cuda.use_mem_pool(token_to_kv_pool.custom_mem_pool)
            if token_to_kv_pool.custom_mem_pool
            else nullcontext()
        )
        with (
            token_to_kv_pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            pool_context,
        ):
            # Fixed address for the lifetime of the runner. Captured
            # draft-extend graphs write it via index_copy_ without rebinding.
            self.activation_buffer = torch.empty(
                (self.num_slots, self.hidden_size),
                dtype=torch.bfloat16,
                device=token_to_kv_pool.device,
            )
            self.position_buffer = torch.empty(
                self.num_slots,
                dtype=torch.int64,
                device=token_to_kv_pool.device,
            )

        self.allocated_bytes = (
            self.activation_buffer.numel() * self.activation_buffer.element_size()
            + self.position_buffer.numel() * self.position_buffer.element_size()
        )
        token_to_kv_pool.mem_usage += self.allocated_bytes / (1024**3)
        attention.diag_es_mtp_kv_replay = self
        logger.info(
            "MTP draft-KV replay buffer is allocated. #tokens: %d, size: %.2f GB",
            self.num_slots,
            self.allocated_bytes / (1024**3),
        )

    def capture(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_mode: ForwardMode,
        out_cache_loc,
    ) -> None:
        """Store candidate-neutral projection inputs at their draft KV slots."""

        if not is_target_driven_mtp_forward(forward_mode):
            return
        if hidden_states.numel() == 0:
            return
        loc, _, _ = unwrap_write_loc(out_cache_loc)
        if loc is None:
            raise RuntimeError("MTP draft-KV replay capture is missing cache locations")
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError("MTP draft-KV replay capture requires BF16 activations")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise RuntimeError(
                "MTP draft-KV replay capture requires [tokens, 2048] activations"
            )
        if loc.ndim != 1 or loc.numel() != hidden_states.shape[0]:
            raise RuntimeError(
                "MTP draft-KV replay capture locations do not match token rows"
            )
        if (
            positions.ndim != 1
            or positions.numel() != hidden_states.shape[0]
            or positions.dtype != torch.int64
        ):
            raise RuntimeError(
                "MTP draft-KV replay capture requires one int64 position per row"
            )
        if loc.dtype != torch.int64:
            raise RuntimeError("MTP draft-KV replay capture locations must be int64")
        self.activation_buffer.index_copy_(0, loc, hidden_states)
        self.position_buffer.index_copy_(0, loc, positions)

    @torch.no_grad()
    def _replay_chunk(
        self,
        *,
        loc: torch.Tensor,
        positions: torch.Tensor,
        candidate_slots: torch.Tensor,
    ) -> None:
        hidden_states = self.activation_buffer.index_select(0, loc)
        # ReplicatedLinear owns the same Triton BF16 pre/post path used by the
        # ordinary model forward. The full fused epilogue is active for post,
        # so the latent slice correctly carries post steering into draft KV.
        with forward_context(
            ForwardContext(attn_backend=None, es_candidate_slots=candidate_slots)
        ):
            qkv_latent = self.fused_projection(hidden_states)[0]

        latent_cache = qkv_latent[:, self.q_lora_rank :]
        k_nope = self.kv_a_layernorm(latent_cache[:, : self.kv_lora_rank]).unsqueeze(1)
        k_pe = latent_cache[:, self.kv_lora_rank :].unsqueeze(1)
        # The regular path rotates q_pe and k_pe together. A one-head dummy q
        # exercises the identical RoPE implementation while avoiding q_b GEMM;
        # q and k have independent head rows, so the resulting k_pe is exact.
        _, k_pe = self.rotary_emb(
            positions,
            torch.zeros_like(k_pe),
            k_pe,
        )
        self.token_to_kv_pool.set_mla_kv_buffer(
            self.attn_mqa,
            loc,
            k_nope,
            k_pe,
        )

    def replay_transitioned_prefixes(
        self,
        batch: ScheduleBatch,
        transitioned: Sequence[bool],
    ) -> MTPKVReplayBatchStats:
        """Rewrite [0:old_seq_len) for requests whose candidate changed."""

        if len(transitioned) != len(batch.reqs):
            raise RuntimeError(
                "MTP draft-KV replay transition mask does not match batch size"
            )
        if not any(transitioned):
            return MTPKVReplayBatchStats(0, 0, 0.0, (0,) * len(transitioned))

        wall_start = time.perf_counter()

        if batch.seq_lens_cpu is None or batch.req_pool_indices_cpu is None:
            raise RuntimeError(
                "MTP draft-KV replay requires verify CPU sequence/pool snapshots"
            )
        seq_lens = [int(value) for value in batch.seq_lens_cpu.tolist()]
        req_pool_indices = [
            int(value) for value in batch.req_pool_indices_cpu.tolist()
        ]
        req_to_token = batch.req_to_token_pool.req_to_token
        total_replayed = 0
        request_rows = [0] * len(batch.reqs)
        loc_parts: list[torch.Tensor] = []
        slot_parts: list[torch.Tensor] = []
        pending_rows = 0

        def flush() -> None:
            nonlocal pending_rows, total_replayed
            if not pending_rows:
                return
            loc = (
                loc_parts[0] if len(loc_parts) == 1 else torch.cat(loc_parts, dim=0)
            ).to(torch.int64)
            positions = self.position_buffer.index_select(0, loc)
            candidate_slots = (
                slot_parts[0] if len(slot_parts) == 1 else torch.cat(slot_parts, dim=0)
            )
            self._replay_chunk(
                loc=loc,
                positions=positions,
                candidate_slots=candidate_slots,
            )
            total_replayed += pending_rows
            loc_parts.clear()
            slot_parts.clear()
            pending_rows = 0

        # Pack rows from multiple switched requests into each replay GEMM. This
        # keeps synchronized B64 transitions to roughly total_rows/chunk_tokens
        # launches instead of one or more launches per request.
        for req_index, changed in enumerate(transitioned):
            if not changed:
                continue
            seq_len = seq_lens[req_index]
            if seq_len < 0:
                raise RuntimeError("MTP draft-KV replay saw a negative sequence length")
            resident_slot = int(batch.reqs[req_index].diag_es_mtp_slot)
            request_rows[req_index] = seq_len
            row = req_to_token[req_pool_indices[req_index]]
            start = 0
            while start < seq_len:
                take = min(seq_len - start, self.chunk_tokens - pending_rows)
                end = start + take
                loc_parts.append(row[start:end])
                slot_parts.append(
                    torch.full(
                        (take,),
                        resident_slot,
                        dtype=torch.int32,
                        device=row.device,
                    )
                )
                pending_rows += take
                start = end
                if pending_rows == self.chunk_tokens:
                    flush()
        flush()
        return MTPKVReplayBatchStats(
            replayed_rows=total_replayed,
            transitioned_requests=sum(bool(value) for value in transitioned),
            enqueue_time_ms=(time.perf_counter() - wall_start) * 1000.0,
            request_rows=tuple(request_rows),
        )
