# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-replay MTP graph recipe.

The regular model runners deliberately stop after the transformer body.  MTP
needs a different lifetime: K draft body/head/sample steps feed one target
verification body/head/sample, followed by acceptance and next-state update.
This module owns that fixed-shape recipe without putting speculative control
flow in a model's ``forward`` method.

The model-specific adapter is supplied as callables.  It is responsible for
installing the correct role-scoped ``ForwardContext`` and for translating the
fixed row layout into the target/draft attention metadata.  The recipe itself
only uses fixed-shape tensors and device operations, so it can be captured by
``torch.npu.NPUGraph`` once the adapter is graph-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.nn as nn

from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
)
from xllm.python.model_executor.runners.base import (
    SpeculativeDeviceState,
    SpeculativeExecutionOutput,
)

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata
    from xllm.python.model_executor.forward_context import LayerSynchronizer

GraphBackend = Literal["eager", "aclgraph"]
ForwardFn = Callable[
    [torch.Tensor, torch.Tensor, int, torch.Tensor | None, torch.Tensor | None],
    torch.Tensor | tuple[torch.Tensor, ...],
]
LogitsFn = Callable[[torch.Tensor], torch.Tensor]
PrepareFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], None]
ActivateFn = Callable[[], None]


class MtpRoleAdapter:
    """Role-scoped body adapter for one composite MTP graph.

    Each role owns a separate ``ModelExecutor`` and therefore a separate
    attention backend, layer-cache list, and ``ForwardContext``.  The adapter
    deliberately calls that executor's eager body runner instead of its decode
    graph runner: the outer MTP runner is the graph being captured, so nesting
    another graph runner here would capture the wrong ownership boundary.
    """

    def __init__(
        self,
        executor: object,
        metadata_by_step: Sequence[AttentionMetadata],
        *,
        speculative_tokens: int,
        target: bool = False,
        layer_synchronizer: LayerSynchronizer | None = None,
        repair_token_ids: torch.Tensor | None = None,
    ) -> None:
        if not metadata_by_step:
            raise ValueError("MTP role adapter requires at least one metadata plan")
        if not hasattr(executor, "eager_runner"):
            raise TypeError("MTP role adapter requires a ModelExecutor-like object")
        if speculative_tokens <= 0:
            raise ValueError("MTP role adapter requires positive fixed K")
        if not target and len(metadata_by_step) != speculative_tokens:
            raise ValueError("draft metadata plan must contain one entry per fixed K step")
        if target and len(metadata_by_step) != 1:
            raise ValueError("target metadata plan must contain one K+1-row entry")
        self._executor = executor
        clone_metadata = getattr(metadata_by_step[0], "clone_for_graph", None)
        if clone_metadata is None:
            self._metadata_by_step = tuple(metadata_by_step)
        else:
            self._metadata_by_step = tuple(item.clone_for_graph() for item in metadata_by_step)
        self._target = target
        self._layer_synchronizer = layer_synchronizer
        self._repair_token_ids = torch.empty_like(repair_token_ids) if repair_token_ids is not None else None
        if self._repair_token_ids is not None:
            self._repair_token_ids.copy_(repair_token_ids)
        self._acl_graph: AclGraphCaptureContext | None = None
        self._execution_state: AclGraphExecutionState | None = None
        create_backends = getattr(executor, "create_mtp_role_backends", None)
        self._attention_backends = (
            tuple(create_backends(len(self._metadata_by_step))) if create_backends is not None else ()
        )
        if self._attention_backends and len(self._attention_backends) != len(self._metadata_by_step):
            raise RuntimeError("MTP role backend count does not match metadata plan")
        self._graph_prepared = False
        self._step_execution_states: tuple[AclGraphExecutionState | None, ...] = ()

    def bind_graph_context(
        self,
        acl_graph: AclGraphCaptureContext | None,
        execution_state: AclGraphExecutionState | None,
    ) -> None:
        """Bind the role-local context owned by the enclosing graph entry."""
        if acl_graph is not None and not self._attention_backends:
            raise RuntimeError("MTP ACL graph requires one pre-prepared attention backend per fixed role metadata plan")
        self._acl_graph = acl_graph
        self._execution_state = execution_state
        if execution_state is None:
            self._step_execution_states = tuple(None for _ in self._metadata_by_step)
        elif not self._step_execution_states:
            # Each fixed invocation owns its graph workspace namespace. The
            # attention backends use generic execution-buffer keys such as
            # FIA_OUTPUT; sharing one map across K steps would alias draft
            # and target workspaces when their shapes happen to match.
            self._step_execution_states = tuple(AclGraphExecutionState({}) for _ in self._metadata_by_step)
        elif len(self._step_execution_states) != len(self._metadata_by_step):
            raise RuntimeError("MTP role execution-state plan changed after warmup")

    def finish_warmup(self) -> None:
        """Mark fixed role backends ready for graph capture and replay."""
        if self._attention_backends:
            self._graph_prepared = True

    def can_update_metadata(self, metadata_by_step: Sequence[AttentionMetadata]) -> bool:
        """Return whether a new invocation fits this captured role variant."""
        if len(metadata_by_step) != len(self._metadata_by_step):
            return False
        can_update = all(hasattr(old_metadata, "update_from") for old_metadata in self._metadata_by_step)
        if not can_update or not self._attention_backends:
            return False
        if not all(hasattr(backend, "update_graph_metadata") for backend in self._attention_backends):
            return False
        for old_metadata, new_metadata in zip(self._metadata_by_step, metadata_by_step):
            if getattr(old_metadata, "is_prefill", None) != getattr(new_metadata, "is_prefill", None):
                return False
            if getattr(old_metadata, "is_chunked_prefill", None) != getattr(new_metadata, "is_chunked_prefill", None):
                return False
            for name in (
                "slot_mapping",
                "paged_kv_indptr",
                "paged_kv_indices",
                "paged_kv_last_page_len",
                "q_cu_seq_lens",
                "kv_cu_seq_lens",
                "kv_seq_lens",
                "q_seq_lens",
                "block_table",
            ):
                old_value = getattr(old_metadata, name, None)
                new_value = getattr(new_metadata, name, None)
                if (old_value is None) != (new_value is None):
                    return False
                if old_value is not None and (
                    old_value.shape != new_value.shape
                    or old_value.dtype != new_value.dtype
                    or old_value.device != new_value.device
                ):
                    return False
            for name in ("kv_seq_lens_host_values", "q_seq_lens_host"):
                old_value = getattr(old_metadata, name, None)
                new_value = getattr(new_metadata, name, None)
                if old_value is None or new_value is None:
                    if old_value is not new_value:
                        return False
                elif len(old_value) != len(new_value):
                    return False
            old_tables = getattr(old_metadata, "multi_block_tables", ())
            new_tables = getattr(new_metadata, "multi_block_tables", ())
            if len(old_tables) != len(new_tables):
                return False
            for old_table, new_table in zip(old_tables, new_tables):
                if (old_table is None) != (new_table is None):
                    return False
                if old_table is not None and (
                    old_table.shape != new_table.shape
                    or old_table.dtype != new_table.dtype
                    or old_table.device != new_table.device
                ):
                    return False
            old_expanded = getattr(old_metadata, "expanded_decode_metadata", None)
            new_expanded = getattr(new_metadata, "expanded_decode_metadata", None)
            old_enabled = bool(getattr(old_expanded, "enabled", False)) if old_expanded is not None else False
            new_enabled = bool(getattr(new_expanded, "enabled", False)) if new_expanded is not None else False
            if old_enabled != new_enabled:
                return False
            if old_enabled:
                for name in (
                    "kv_seq_lens",
                    "block_table",
                    "paged_kv_indptr",
                    "paged_kv_indices",
                    "paged_kv_last_page_len",
                    "paged_attention_tiling_data",
                    "kv_seq_lens_host",
                ):
                    old_value = getattr(old_expanded, name, None)
                    new_value = getattr(new_expanded, name, None)
                    if (old_value is None) != (new_value is None):
                        return False
                    if old_value is not None and (
                        old_value.shape != new_value.shape
                        or old_value.dtype != new_value.dtype
                        or old_value.device != new_value.device
                    ):
                        return False
                old_host_values = getattr(old_expanded, "kv_seq_lens_host_values", None)
                new_host_values = getattr(new_expanded, "kv_seq_lens_host_values", None)
                if (old_host_values is None) != (new_host_values is None):
                    return False
                if old_host_values is not None and len(old_host_values) != len(new_host_values):
                    return False
        return True

    @torch.inference_mode()
    def update_metadata(
        self,
        metadata_by_step: Sequence[AttentionMetadata],
        repair_token_ids: torch.Tensor | None,
    ) -> None:
        """Copy a new invocation into the captured role's stable metadata."""
        if not self._graph_prepared:
            raise RuntimeError("MTP role metadata cannot update before graph warmup")
        if not self.can_update_metadata(metadata_by_step):
            raise RuntimeError("MTP role metadata is not compatible with captured graph variant")
        if (self._repair_token_ids is None) != (repair_token_ids is None):
            raise RuntimeError("MTP repair-token presence changed after capture")
        if self._repair_token_ids is not None:
            assert repair_token_ids is not None
            if repair_token_ids.shape != self._repair_token_ids.shape:
                raise RuntimeError("MTP repair-token shape changed after capture")
            self._repair_token_ids.copy_(repair_token_ids)
        for old_metadata, new_metadata, backend in zip(
            self._metadata_by_step,
            metadata_by_step,
            self._attention_backends,
        ):
            old_metadata.update_from(new_metadata)
            update_backend = getattr(backend, "update_graph_metadata", None)
            if update_backend is None:
                raise RuntimeError("MTP graph attention backend does not support metadata replay updates")
            update_backend(old_metadata)

    def __call__(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        metadata_index = 0 if self._target else step
        if metadata_index < 0 or metadata_index >= len(self._metadata_by_step):
            raise ValueError(f"MTP role adapter received invalid step={step}")
        call_token_ids = token_ids
        call_positions = positions
        call_embedding = input_embedding
        selected_rows: torch.Tensor | None = None
        if not self._target and step == 0 and self._repair_token_ids is not None:
            if self._repair_token_ids.shape != token_ids.shape:
                raise ValueError("MTP repair token ids must match the draft batch")
            if input_embedding is None or input_embedding.shape[0] != token_ids.numel() * 2:
                raise ValueError("MTP first draft repair layout requires 2B embeddings")
            call_token_ids = torch.stack((self._repair_token_ids, token_ids), dim=1).reshape(-1)
            call_positions = torch.stack((positions - 1, positions), dim=1).reshape(-1)
            selected_rows = torch.arange(
                1,
                token_ids.numel() * 2,
                2,
                dtype=torch.long,
                device=token_ids.device,
            )
        execute_role = getattr(self._executor, "execute_mtp_role", None)
        if execute_role is None:
            if self._acl_graph is not None or self._execution_state is not None:
                raise TypeError("MTP graph adapter executor lacks execute_mtp_role")
            output = self._executor.eager_runner.execute(
                call_token_ids,
                call_positions,
                self._metadata_by_step[metadata_index],
                call_embedding,
                self._layer_synchronizer,
                topk_indices,
            )
        else:
            execution_state = (
                self._step_execution_states[metadata_index] if self._step_execution_states else self._execution_state
            )
            output = execute_role(
                call_token_ids,
                call_positions,
                self._metadata_by_step[metadata_index],
                call_embedding,
                self._layer_synchronizer,
                topk_indices,
                acl_graph=self._acl_graph,
                execution_state=execution_state,
                attention_backend=(self._attention_backends[metadata_index] if self._attention_backends else None),
                skip_prepare=self._graph_prepared,
            )
        if selected_rows is None:
            return output
        if isinstance(output, tuple):
            hidden = output[0].index_select(0, selected_rows)
            if len(output) >= 3 and isinstance(output[2], torch.Tensor):
                return hidden, output[1], output[2].index_select(0, selected_rows)
            return hidden, output[1]
        return output.index_select(0, selected_rows)


@dataclass(frozen=True)
class MtpGraphCapability:
    """Static admission result for one MTP graph variant."""

    speculative_tokens: int
    batch_size: int
    vocab_size: int
    sampling_mode: str = "greedy"
    supports_aclgraph: bool = True
    reason: str | None = None

    def require_aclgraph(self) -> None:
        if not self.supports_aclgraph:
            raise RuntimeError(self.reason or "MTP ACL graph variant is not supported")
        if self.sampling_mode != "greedy":
            raise NotImplementedError("the first fused MTP graph only supports greedy sampling")


@dataclass
class MtpGraphEntry:
    """Lifecycle record for one fixed-shape graph variant."""

    capability: MtpGraphCapability
    generation: int = 0
    captured: bool = False
    graph: object | None = None


@dataclass(frozen=True)
class MtpGraphOutput:
    """Persistent output views returned by the recipe.

    ``accepted_ids`` contains only the accepted draft prefix and is padded
    with ``-1``.  ``next_tokens`` contains the target replacement token after
    the first rejection, or the target bonus token when all K drafts match.
    ``accepted_count`` therefore counts draft tokens only; the caller emits
    one additional ``next_tokens`` value.
    """

    accepted_ids: torch.Tensor
    accepted_mask: torch.Tensor
    accepted_count: torch.Tensor
    next_tokens: torch.Tensor
    committed_tokens: torch.Tensor
    draft_tokens: torch.Tensor
    target_tokens: torch.Tensor
    next_positions: torch.Tensor
    next_kv_seq_lens: torch.Tensor | None
    next_embeddings: torch.Tensor | None
    next_topk_indices: torch.Tensor | None
    target_embeddings: torch.Tensor


@dataclass(frozen=True)
class MtpBodyOutput:
    """Body result needed by the next speculative invocation."""

    hidden: torch.Tensor
    topk_indices: torch.Tensor | None = None


def _body_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> MtpBodyOutput:
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError("MTP model adapter must return a tensor or a tensor-first tuple")
        topk_indices = output[2] if len(output) >= 3 and isinstance(output[2], torch.Tensor) else None
        return MtpBodyOutput(output[0], topk_indices)
    return MtpBodyOutput(output)


def greedy_acceptance(
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a fixed-shape MTP accepted prefix on Device.

    The target emits K+1 rows.  For a rejection at row ``j``, the target token
    at row ``j`` is the replacement and later draft rows are discarded.  When
    every draft matches, the final target row is the bonus token.  ``argmax``
    and ``cumprod`` keep the shape fixed for all rejection positions.
    """

    if draft_tokens.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError("MTP token matrices must have shape [batch, steps]")
    batch_size, speculative_tokens = draft_tokens.shape
    if target_tokens.shape != (batch_size, speculative_tokens + 1):
        raise ValueError("target token matrix must contain K+1 rows per request")
    if speculative_tokens == 0:
        raise ValueError("MTP acceptance requires at least one draft token")

    matches = draft_tokens.eq(target_tokens[:, :speculative_tokens])
    accepted_mask = torch.cumprod(matches.to(torch.int32), dim=1).to(torch.bool)
    accepted_count = accepted_mask.sum(dim=1, dtype=torch.int32)

    rejected = ~accepted_mask
    # For an all-accepted row argmax returns zero; select the final bonus in
    # that case.  The gather remains fixed-shape and does not read host state.
    first_rejection = rejected.to(torch.int32).argmax(dim=1)
    replacement = target_tokens[:, :speculative_tokens].gather(1, first_rejection.unsqueeze(1)).squeeze(1)
    all_accepted = accepted_count.eq(speculative_tokens)
    bonus = target_tokens[:, speculative_tokens]
    next_tokens = torch.where(all_accepted, bonus, replacement)
    accepted_ids = torch.where(
        accepted_mask,
        draft_tokens,
        torch.full_like(draft_tokens, -1),
    )
    return accepted_ids, accepted_mask, accepted_count, next_tokens


def _committed_tokens(
    accepted_ids: torch.Tensor,
    accepted_count: torch.Tensor,
    next_tokens: torch.Tensor,
    speculative_tokens: int,
) -> torch.Tensor:
    """Materialize the runtime's fixed-width accepted/replacement layout."""

    all_accepted = accepted_count.eq(speculative_tokens)
    positions = torch.arange(
        speculative_tokens,
        dtype=accepted_count.dtype,
        device=accepted_count.device,
    ).unsqueeze(0)
    first_rejection = accepted_count.unsqueeze(1)
    replacement_mask = positions.eq(first_rejection) & ~all_accepted.unsqueeze(1)
    draft_rows = torch.where(
        replacement_mask,
        next_tokens.unsqueeze(1),
        accepted_ids,
    )
    bonus = torch.where(
        all_accepted,
        next_tokens,
        torch.full_like(next_tokens, -1),
    )
    return torch.cat((draft_rows, bonus.unsqueeze(1)), dim=1)


class MtpGraphRecipe(nn.Module):
    """K-unrolled draft/target recipe used by eager oracle and ACL graph."""

    def __init__(
        self,
        draft_forward: ForwardFn,
        draft_logits: LogitsFn,
        target_forward: ForwardFn,
        target_logits: LogitsFn,
        *,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        device: torch.device,
        kv_seq_lens: torch.Tensor | None = None,
        draft_activate: ActivateFn | None = None,
        target_activate: ActivateFn | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if speculative_tokens <= 0:
            raise ValueError("speculative_tokens must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if kv_seq_lens is not None and tuple(kv_seq_lens.shape) != (batch_size,):
            raise ValueError("kv_seq_lens must have one value per request")
        self.batch_size = batch_size
        self.speculative_tokens = speculative_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.draft_forward = draft_forward
        self.draft_logits = draft_logits
        self.target_forward = target_forward
        self.target_logits = target_logits
        self.draft_activate = draft_activate
        self.target_activate = target_activate
        self.register_buffer(
            "_kv_seq_lens_template",
            kv_seq_lens.to(device=device, dtype=torch.int32).contiguous()
            if kv_seq_lens is not None
            else torch.empty(0, dtype=torch.int32, device=device),
            persistent=False,
        )

    def bind_graph_contexts(
        self,
        acl_graph: AclGraphCaptureContext | None,
        draft_state: AclGraphExecutionState | None,
        target_state: AclGraphExecutionState | None,
    ) -> None:
        """Bind role-local graph state without assuming adapter internals."""
        for forward, state in (
            (self.draft_forward, draft_state),
            (self.target_forward, target_state),
        ):
            bind_context = getattr(forward, "bind_graph_context", None)
            if bind_context is not None:
                bind_context(acl_graph, state)

    def finish_warmup(self) -> None:
        """Freeze role backend metadata before entering ACL graph capture."""
        for forward in (self.draft_forward, self.target_forward):
            finish_warmup = getattr(forward, "finish_warmup", None)
            if finish_warmup is not None:
                finish_warmup()

    def can_update_metadata(
        self,
        draft_metadata: Sequence[AttentionMetadata],
        target_metadata: AttentionMetadata,
    ) -> bool:
        """Check whether both role adapters can consume a new invocation."""
        draft_check = getattr(self.draft_forward, "can_update_metadata", None)
        target_check = getattr(self.target_forward, "can_update_metadata", None)
        if draft_check is None or target_check is None:
            return False
        return bool(draft_check(draft_metadata)) and bool(target_check((target_metadata,)))

    def update_metadata(
        self,
        draft_metadata: Sequence[AttentionMetadata],
        target_metadata: AttentionMetadata,
        repair_token_ids: torch.Tensor,
    ) -> None:
        """Update both role metadata and the captured repair-token buffer."""
        update_draft = getattr(self.draft_forward, "update_metadata", None)
        update_target = getattr(self.target_forward, "update_metadata", None)
        if update_draft is None or update_target is None:
            raise RuntimeError("MTP graph recipe does not support metadata replay updates")
        update_draft(draft_metadata, repair_token_ids)
        update_target((target_metadata,), None)

    @torch.inference_mode()
    def forward(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None = None,
        draft_input_embedding: torch.Tensor | None = None,
        draft_topk_indices: torch.Tensor | None = None,
    ) -> MtpGraphOutput:
        if seed_token_ids.shape != (self.batch_size,):
            raise ValueError("seed_token_ids shape does not match the captured batch")
        if base_positions.shape != (self.batch_size,):
            raise ValueError("base_positions shape does not match the captured batch")
        if kv_seq_lens is not None and kv_seq_lens.shape != (self.batch_size,):
            raise ValueError("kv_seq_lens shape does not match the captured batch")

        draft_tokens: list[torch.Tensor] = []
        current_ids = seed_token_ids
        current_embedding = draft_input_embedding
        current_topk_indices = draft_topk_indices
        for step in range(self.speculative_tokens):
            if self.draft_activate is not None:
                self.draft_activate()
            step_positions = base_positions + step
            try:
                draft_output = _body_output(
                    self.draft_forward(
                        current_ids,
                        step_positions,
                        step,
                        current_embedding,
                        current_topk_indices,
                    )
                )
            except Exception as exc:
                raise RuntimeError(f"MTP draft body step {step} failed: {exc}") from exc
            draft_hidden = draft_output.hidden
            step_logits = self.draft_logits(draft_hidden)
            if step_logits.shape != (self.batch_size, self.vocab_size):
                raise ValueError("draft logits must have shape [batch, vocab]")
            current_ids = step_logits.argmax(dim=-1)
            # GLM MTP consumes the preceding body hidden as the next-step
            # embedding and optionally reuses the preceding DSA top-k state.
            current_embedding = draft_hidden
            current_topk_indices = draft_output.topk_indices
            draft_tokens.append(current_ids)
        draft_matrix = torch.stack(draft_tokens, dim=1)

        target_token_matrix = torch.cat((seed_token_ids.unsqueeze(1), draft_matrix), dim=1)
        target_ids = target_token_matrix.reshape(-1)
        target_positions = (
            base_positions.unsqueeze(1)
            + torch.arange(
                self.speculative_tokens + 1,
                dtype=base_positions.dtype,
                device=base_positions.device,
            ).unsqueeze(0)
        ).reshape(-1)
        if self.target_activate is not None:
            self.target_activate()
        try:
            target_output = _body_output(self.target_forward(target_ids, target_positions, -1, None, None))
        except Exception as exc:
            raise RuntimeError(f"MTP target body failed: {exc}") from exc
        target_hidden = target_output.hidden
        if target_hidden.ndim != 2 or target_hidden.shape[0] != self.batch_size * (self.speculative_tokens + 1):
            raise ValueError("target hidden must have one row per target verification token")
        target_logits = self.target_logits(target_hidden)
        if target_logits.shape != (
            self.batch_size * (self.speculative_tokens + 1),
            self.vocab_size,
        ):
            raise ValueError("target logits must have shape [batch*(K+1), vocab]")
        target_tokens = target_logits.argmax(dim=-1).reshape(self.batch_size, self.speculative_tokens + 1)

        (
            accepted_ids,
            accepted_mask,
            accepted_count,
            next_tokens,
        ) = greedy_acceptance(draft_matrix, target_tokens)
        committed_tokens = _committed_tokens(
            accepted_ids,
            accepted_count,
            next_tokens,
            self.speculative_tokens,
        )
        # ``next_tokens`` is the replacement token after a rejection or the
        # bonus token after an all-accepted verify.  It is emitted in addition
        # to the accepted draft prefix, so the next decode position advances
        # by accepted draft tokens plus one emitted target token.
        emitted_length = accepted_count.to(base_positions.dtype) + 1
        next_positions = base_positions + emitted_length
        next_kv_seq_lens = None
        if kv_seq_lens is not None:
            next_kv_seq_lens = kv_seq_lens + emitted_length.to(kv_seq_lens.dtype)
        target_hidden_matrix = target_hidden.reshape(
            self.batch_size,
            self.speculative_tokens + 1,
            target_hidden.shape[-1],
        )
        next_embedding_indices = accepted_count.to(torch.long).view(self.batch_size, 1, 1)
        next_embedding_indices = next_embedding_indices.expand(-1, 1, target_hidden_matrix.shape[-1])
        next_embeddings = target_hidden_matrix.gather(1, next_embedding_indices).squeeze(1)
        next_topk_indices = None
        if target_output.topk_indices is not None:
            target_topk = target_output.topk_indices
            if target_topk.shape[0] != self.batch_size * (self.speculative_tokens + 1):
                raise ValueError("target top-k state must have one row per target verification token")
            target_topk_matrix = target_topk.reshape(
                self.batch_size,
                self.speculative_tokens + 1,
                *target_topk.shape[1:],
            )
            topk_index = accepted_count.to(torch.long).reshape(self.batch_size, 1)
            topk_index = topk_index.reshape(self.batch_size, 1, *([1] * (target_topk_matrix.ndim - 2)))
            topk_index = topk_index.expand(-1, 1, *target_topk_matrix.shape[2:])
            next_topk_indices = target_topk_matrix.gather(1, topk_index).squeeze(1)
        return MtpGraphOutput(
            accepted_ids=accepted_ids,
            accepted_mask=accepted_mask,
            accepted_count=accepted_count,
            next_tokens=next_tokens,
            committed_tokens=committed_tokens,
            draft_tokens=draft_matrix,
            target_tokens=target_tokens,
            next_positions=next_positions,
            next_kv_seq_lens=next_kv_seq_lens,
            next_embeddings=next_embeddings,
            next_topk_indices=next_topk_indices,
            target_embeddings=target_hidden_matrix,
        )


class MtpAclGraphRunner:
    """Capture and replay one fixed-shape MTP recipe.

    The runner intentionally requires a graph-safe model adapter.  It does
    not catch capture errors and silently switch to eager execution; callers
    choose ``backend='eager'`` explicitly for the oracle/fallback path.
    """

    def __init__(
        self,
        recipe: MtpGraphRecipe,
        *,
        backend: GraphBackend = "aclgraph",
        warmup_steps: int = 2,
        prepare: PrepareFn | None = None,
    ) -> None:
        if backend not in ("eager", "aclgraph"):
            raise ValueError(f"unknown MTP graph backend: {backend!r}")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        self.recipe = recipe
        self.backend = backend
        self.warmup_steps = warmup_steps
        self.prepare = prepare
        self._captured = False
        self._graph = None
        self._capture_stream = None
        self._execution_states = (
            (AclGraphExecutionState({}), AclGraphExecutionState({})) if backend == "aclgraph" else (None, None)
        )
        self._entry = MtpGraphEntry(
            MtpGraphCapability(
                speculative_tokens=recipe.speculative_tokens,
                batch_size=recipe.batch_size,
                vocab_size=recipe.vocab_size,
                supports_aclgraph=backend == "aclgraph",
                reason="runner was created with the explicit eager backend" if backend != "aclgraph" else None,
            )
        )
        self._static_seed_token_ids = torch.zeros(recipe.batch_size, dtype=torch.long, device=recipe.device)
        self._static_base_positions = torch.zeros(recipe.batch_size, dtype=torch.long, device=recipe.device)
        self._static_kv_seq_lens = (
            torch.zeros(recipe.batch_size, dtype=torch.int32, device=recipe.device)
            if recipe._kv_seq_lens_template.numel()
            else None
        )
        self._static_draft_input_embedding: torch.Tensor | None = None
        self._static_draft_topk_indices: torch.Tensor | None = None
        self._static_output: MtpGraphOutput | None = None

    def _prepare_static_inputs(self) -> None:
        if self.prepare is not None:
            self.prepare(
                self._static_seed_token_ids,
                self._static_base_positions,
                self._static_kv_seq_lens,
            )

    @property
    def capability(self) -> MtpGraphCapability:
        return self._entry.capability

    @property
    def entry(self) -> MtpGraphEntry:
        return self._entry

    @property
    def captured(self) -> bool:
        return self._captured

    def _validate_inputs(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None,
    ) -> None:
        expected_device = self.recipe.device
        for name, tensor in (("seed_token_ids", seed_token_ids), ("base_positions", base_positions)):
            if tensor.device != expected_device or tensor.shape != (self.recipe.batch_size,):
                raise ValueError(f"{name} must be a [{self.recipe.batch_size}] tensor on {expected_device}")
        if seed_token_ids.dtype != torch.long or base_positions.dtype != torch.long:
            raise ValueError("MTP token ids and positions must use int64 tensors")
        if self._static_kv_seq_lens is not None:
            if kv_seq_lens is None or kv_seq_lens.device != expected_device:
                raise ValueError("kv_seq_lens is required for this graph variant")
            if kv_seq_lens.shape != (self.recipe.batch_size,) or kv_seq_lens.dtype != torch.int32:
                raise ValueError("kv_seq_lens must be an int32 vector matching the graph batch")
        elif kv_seq_lens is not None:
            raise ValueError("this graph variant was captured without kv_seq_lens")

    def _validate_recurrent_inputs(
        self,
        draft_input_embedding: torch.Tensor | None,
        draft_topk_indices: torch.Tensor | None,
    ) -> None:
        if self.backend == "eager":
            return
        expected_device = self.recipe.device
        for name, value, static in (
            ("draft_input_embedding", draft_input_embedding, self._static_draft_input_embedding),
            ("draft_topk_indices", draft_topk_indices, self._static_draft_topk_indices),
        ):
            if static is None:
                if value is not None:
                    raise ValueError(f"{name} was not part of the captured graph variant")
                continue
            if value is None:
                raise ValueError(f"{name} is required for this graph variant")
            if value.device != expected_device or value.shape != static.shape or value.dtype != static.dtype:
                raise ValueError(f"{name} shape, dtype, or device changed after graph capture")

    def _run_static(self) -> MtpGraphOutput:
        return self.recipe(
            self._static_seed_token_ids,
            self._static_base_positions,
            self._static_kv_seq_lens,
            self._static_draft_input_embedding,
            self._static_draft_topk_indices,
        )

    def can_update_metadata(
        self,
        draft_metadata: Sequence[AttentionMetadata],
        target_metadata: AttentionMetadata,
    ) -> bool:
        """Return whether the captured graph can reuse stable metadata storage."""
        if not self._captured:
            return False
        return self.recipe.can_update_metadata(draft_metadata, target_metadata)

    def update_metadata(
        self,
        draft_metadata: Sequence[AttentionMetadata],
        target_metadata: AttentionMetadata,
        repair_token_ids: torch.Tensor,
    ) -> None:
        """Update role-local metadata before the next graph replay."""
        if not self._captured:
            raise RuntimeError("MTP ACL graph must be captured before metadata update")
        self.recipe.update_metadata(draft_metadata, target_metadata, repair_token_ids)

    def capture(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None = None,
        draft_input_embedding: torch.Tensor | None = None,
        draft_topk_indices: torch.Tensor | None = None,
    ) -> None:
        self.capability.require_aclgraph()
        self._validate_inputs(seed_token_ids, base_positions, kv_seq_lens)
        if draft_input_embedding is not None:
            self._static_draft_input_embedding = torch.empty_like(draft_input_embedding)
        if draft_topk_indices is not None:
            self._static_draft_topk_indices = torch.empty_like(draft_topk_indices)
        self._validate_recurrent_inputs(draft_input_embedding, draft_topk_indices)
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            raise RuntimeError("ACL graph backend requires torch.npu.NPUGraph")
        stage = "input copy"
        try:
            self._static_seed_token_ids.copy_(seed_token_ids)
            self._static_base_positions.copy_(base_positions)
            if self._static_kv_seq_lens is not None:
                assert kv_seq_lens is not None
                self._static_kv_seq_lens.copy_(kv_seq_lens)
            if self._static_draft_input_embedding is not None:
                assert draft_input_embedding is not None
                self._static_draft_input_embedding.copy_(draft_input_embedding)
            if self._static_draft_topk_indices is not None:
                assert draft_topk_indices is not None
                self._static_draft_topk_indices.copy_(draft_topk_indices)

            stage = "warmup stream"
            self._capture_stream = torch.npu.Stream(device=self.recipe.device)
            self._capture_stream.wait_stream(torch.npu.current_stream())
            self.recipe.bind_graph_contexts(None, *self._execution_states)
            stage = "warmup recipe"
            with torch.npu.stream(self._capture_stream):
                for _ in range(self.warmup_steps):
                    self._prepare_static_inputs()
                    self._run_static()
            self.recipe.finish_warmup()
            stage = "warmup synchronize"
            self._capture_stream.synchronize()
            torch.npu.current_stream().wait_stream(self._capture_stream)
            torch.npu.synchronize()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                stage = "warmup collective barrier"
                torch.distributed.barrier()

            stage = "graph capture"
            self._graph = torch.npu.NPUGraph()
            capture_context = AclGraphCaptureContext(self._capture_stream, [])
            self.recipe.bind_graph_contexts(capture_context, *self._execution_states)
            with torch.npu.stream(self._capture_stream), torch.npu.graph(self._graph, stream=self._capture_stream):
                self._static_output = self._run_static()
        except Exception as exc:
            raise RuntimeError(f"MTP ACL graph capture failed during {stage}: {exc}") from exc
        self._entry.graph = self._graph
        self._entry.generation += 1
        self._entry.captured = True
        self._captured = True

    def execute(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None = None,
        draft_input_embedding: torch.Tensor | None = None,
        draft_topk_indices: torch.Tensor | None = None,
    ) -> SpeculativeExecutionOutput:
        self._validate_inputs(seed_token_ids, base_positions, kv_seq_lens)
        self._validate_recurrent_inputs(draft_input_embedding, draft_topk_indices)
        if self.backend == "eager":
            if self.prepare is not None:
                self.prepare(seed_token_ids, base_positions, kv_seq_lens)
            output = self.recipe(
                seed_token_ids,
                base_positions,
                kv_seq_lens,
                draft_input_embedding,
                draft_topk_indices,
            )
        else:
            if not self._captured or self._graph is None or self._capture_stream is None:
                raise RuntimeError("MTP ACL graph must be captured before replay")
            self._static_seed_token_ids.copy_(seed_token_ids)
            self._static_base_positions.copy_(base_positions)
            if self._static_kv_seq_lens is not None:
                assert kv_seq_lens is not None
                self._static_kv_seq_lens.copy_(kv_seq_lens)
            if self._static_draft_input_embedding is not None:
                assert draft_input_embedding is not None
                self._static_draft_input_embedding.copy_(draft_input_embedding)
            if self._static_draft_topk_indices is not None:
                assert draft_topk_indices is not None
                self._static_draft_topk_indices.copy_(draft_topk_indices)
            self._prepare_static_inputs()
            self._capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(self._capture_stream):
                self._graph.replay()
            torch.npu.current_stream().wait_stream(self._capture_stream)
            assert self._static_output is not None
            output = self._static_output

        next_state = SpeculativeDeviceState(
            token_ids=output.next_tokens,
            positions=output.next_positions,
            kv_seq_lens=output.next_kv_seq_lens,
            embeddings=output.next_embeddings,
            cache_slots=None,
            topk_indices=output.next_topk_indices,
            committed_mask=output.accepted_mask,
        )
        return SpeculativeExecutionOutput(
            accepted_ids=output.accepted_ids,
            accepted_mask=output.accepted_mask,
            accepted_count=output.accepted_count,
            next_state=next_state,
            committed_tokens=output.committed_tokens,
            target_embeddings=output.target_embeddings,
            draft_tokens=output.draft_tokens,
            target_tokens=output.target_tokens,
        )
