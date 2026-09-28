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

import os
from collections.abc import Sequence, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    AclGraphTask,
)
from xllm.python.model_executor.runners.base import (
    SpeculativeDeviceState,
    SpeculativeExecutionOutput,
    SpeculativeRuntimeOutput,
)
from xllm.python.model_executor.runners.mtp_sampling import (
    MtpSamplingPlan,
    MtpSamplingRandomInputs,
    coerce_sampling_plan,
    probabilistic_acceptance,
    sample_logits,
)
from xllm.python.model_executor.runners.mtp_sparse_metadata import (
    MtpSparseMetadataBinding,
    MtpSparseMetadataStorage,
    MtpSparsePositionStorage,
    SparseMetadataFactory,
)

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata
    from xllm.python.model_executor.forward_context import LayerSynchronizer
    from xllm.python.model_executor.runners.mtp_kv_oracle import MtpKvPayloadOracle

GraphBackend = Literal["eager", "aclgraph"]
ForwardFn = Callable[
    [torch.Tensor, torch.Tensor, int, torch.Tensor | None, torch.Tensor | None],
    torch.Tensor | tuple[torch.Tensor, ...],
]
LogitsFn = Callable[[torch.Tensor], torch.Tensor]
PrepareFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], None]
ActivateFn = Callable[[], None]


@torch.inference_mode()
def _pack_metadata_tensors(metadata_by_step: Sequence[object]) -> tuple[torch.Tensor, ...]:
    """Pack owned metadata into aligned arenas before any backend captures it.

    Rebind the tensor objects themselves, so read-only native metadata views
    and their update_from implementation continue to use the same TensorImpl.
    Each field/step gets its own slice; no replay reads a sibling step's value.
    """
    fields = (
        "slot_mapping",
        "local_slot_mapping",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "qo_indptr",
        "q_cu_seq_lens",
        "kv_cu_seq_lens",
        "kv_seq_lens",
        "q_seq_lens",
        "block_table",
        "kv_seq_lens_host",
        "q_seq_lens_host",
        "linear_state_indices",
        "has_initial_state",
        "paged_attention_tiling_data",
        "dsa_positions",
        "dsa_cos_sin",
        "dsa_c4_cos_sin",
        "dsa_c128_cos_sin",
    )
    groups: dict[tuple[torch.device, torch.dtype, bool], list[torch.Tensor]] = {}
    seen: set[int] = set()
    for metadata in metadata_by_step:
        expanded = getattr(metadata, "expanded_decode_metadata", None)
        for view in (metadata, expanded):
            if view is None:
                continue
            values = [getattr(view, name, None) for name in fields]
            values.extend(getattr(view, "multi_block_tables", ()))
            for value in values:
                if not isinstance(value, torch.Tensor) or value.numel() == 0:
                    continue
                # Native getters may create different Python wrappers for the
                # same TensorImpl. Rebinding that object twice would orphan a slice.
                identity = value._cdata
                if identity in seen:
                    continue
                seen.add(identity)
                pinned = value.device.type == "cpu" and value.is_pinned()
                groups.setdefault((value.device, value.dtype, pinned), []).append(value)
    arenas: list[torch.Tensor] = []
    for (device, dtype, pinned), tensors in groups.items():
        alignment = max(1, 512 // tensors[0].element_size())
        sizes = [((value.numel() + alignment - 1) // alignment) * alignment for value in tensors]
        arena = torch.empty(sum(sizes), dtype=dtype, device=device, pin_memory=pinned)
        offset = 0
        for value, size in zip(tensors, sizes):
            destination = arena.narrow(0, offset, value.numel()).view(value.shape)
            destination.copy_(value)
            value.set_(destination)
            offset += size
        arenas.append(arena)
    return tuple(arenas)


def _tensor_signature(value: object) -> tuple[object, ...] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is None or dtype is None or device is None:
        return (type(value).__qualname__, repr(value))
    return (tuple(shape), str(dtype), str(device))


def _metadata_signature(metadata: AttentionMetadata) -> tuple[object, ...]:
    fields = (
        "slot_mapping",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "q_cu_seq_lens",
        "kv_cu_seq_lens",
        "kv_seq_lens",
        "q_seq_lens",
        "block_table",
    )
    tensor_fields = tuple((name, _tensor_signature(getattr(metadata, name, None))) for name in fields)
    host_fields = tuple(
        (
            name,
            len(getattr(metadata, name, ())) if getattr(metadata, name, None) is not None else None,
        )
        for name in ("kv_seq_lens_host_values", "q_seq_lens_host")
    )
    tables = tuple(_tensor_signature(table) for table in getattr(metadata, "multi_block_tables", ()))
    expanded = getattr(metadata, "expanded_decode_metadata", None)
    expanded_fields = (
        "kv_seq_lens",
        "block_table",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "paged_attention_tiling_data",
        "kv_seq_lens_host",
    )
    expanded_signature = (
        False,
        (),
        None,
    )
    if expanded is not None and bool(getattr(expanded, "enabled", False)):
        expanded_signature = (
            True,
            tuple((name, _tensor_signature(getattr(expanded, name, None))) for name in expanded_fields),
            len(getattr(expanded, "kv_seq_lens_host_values", ()))
            if getattr(expanded, "kv_seq_lens_host_values", None) is not None
            else None,
        )
    return (
        bool(getattr(metadata, "is_prefill", False)),
        bool(getattr(metadata, "is_chunked_prefill", False)),
        bool(getattr(metadata, "is_mixed", False)),
        bool(getattr(metadata, "is_spec_verify", False)),
        tensor_fields,
        host_fields,
        tables,
        expanded_signature,
    )


def _normalize_request_major_rows(
    values: torch.Tensor,
    batch_size: int,
    row_width: int,
    *,
    step_major_layout: bool,
) -> torch.Tensor:
    """Normalize sequence-major or step-major rows to ``[B, row_width]``."""
    if batch_size <= 0 or row_width <= 0:
        raise ValueError("row normalization requires positive batch and width")
    flat = values.reshape(-1)
    expected = batch_size * row_width
    if flat.numel() != expected:
        raise ValueError(f"row layout has {flat.numel()} values, expected {expected}")
    if step_major_layout:
        return flat.view(row_width, batch_size).transpose(0, 1).contiguous()
    return flat.view(batch_size, row_width)


def _metadata_values_empty(values: object) -> bool:
    if values is None:
        return True
    if isinstance(values, torch.Tensor):
        return values.numel() == 0
    return isinstance(values, Sized) and len(values) == 0


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
        step_major_layout: bool = False,
        layer_synchronizer: LayerSynchronizer | None = None,
        repair_token_ids: torch.Tensor | None = None,
        metadata_storage: MtpSparseMetadataStorage | None = None,
        repair_positions: torch.Tensor | None = None,
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
        if metadata_storage is not None:
            if len(metadata_by_step) != len(metadata_storage.metadata) or any(
                actual is not owned for actual, owned in zip(metadata_by_step, metadata_storage.metadata)
            ):
                raise ValueError("sparse MTP role metadata must belong to the supplied storage")
            self._metadata_by_step = metadata_storage.metadata
            self._metadata_arenas = (metadata_storage.arena,)
        elif clone_metadata is None:
            self._metadata_by_step = tuple(metadata_by_step)
            self._metadata_arenas = ()
        else:
            self._metadata_by_step = tuple(item.clone_for_graph() for item in metadata_by_step)
            # Only clones may be rebound. Sparse storage already owns final
            # slices; generic caller metadata retains its original TensorImpl.
            self._metadata_arenas = _pack_metadata_tensors(self._metadata_by_step)
        self._target = target
        self._step_major_layout = step_major_layout
        self._layer_synchronizer = layer_synchronizer
        self._repair_positions = repair_positions
        # Worker may supply a strided int32 view of its two-row fused output.
        # Cast while copying into the final graph-owned int64 destination.
        self._repair_token_ids = (
            torch.empty(repair_token_ids.shape, dtype=torch.long, device=repair_token_ids.device)
            if repair_token_ids is not None
            else None
        )
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
        self._shared_execution_buffers: dict[tuple[object, ...], object] | None = None

    def create_eager_reference(self) -> MtpRoleAdapter:
        """Create an independent backend/metadata context for diagnostics."""
        return MtpRoleAdapter(
            self._executor,
            self._metadata_by_step,
            speculative_tokens=len(self._metadata_by_step),
            target=self._target,
            step_major_layout=self._step_major_layout,
            layer_synchronizer=self._layer_synchronizer,
            repair_token_ids=self._repair_token_ids,
        )

    def cache_write_slots(self) -> torch.Tensor:
        """Return every planned write, including draft repair rows."""
        return torch.cat([metadata.slot_mapping.reshape(-1) for metadata in self._metadata_by_step])

    def verify_slots(self, batch_size: int) -> torch.Tensor:
        """Return target verify writes in request-major rows for diagnostics."""
        if not self._target:
            raise RuntimeError("only the target role has verify slots")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        slot_mapping = self._metadata_by_step[0].slot_mapping.reshape(-1)
        if slot_mapping.numel() % batch_size != 0:
            raise RuntimeError("target verify slot mapping cannot be divided by graph batch size")
        verify_width = slot_mapping.numel() // batch_size
        return _normalize_request_major_rows(
            slot_mapping,
            batch_size,
            verify_width,
            step_major_layout=self._step_major_layout,
        )

    def attention_read_plan(self, batch_size: int) -> tuple[torch.Tensor, object, int]:
        """Expose the backend's current read table and KV lengths to the oracle."""
        if not self._target:
            raise RuntimeError("only the target role has an attention read plan")
        if not self._attention_backends:
            raise RuntimeError("attention read observation requires prepared role backends")
        metadata = self._metadata_by_step[0]
        backend = self._attention_backends[0]
        # The backend resolves expanded decode/DCP layout before execution.
        # Observe that resolved table first; generic metadata can describe a
        # logical table that is not the table consumed by the attention call.
        block_table = getattr(backend, "_block_table_i32", None)
        if block_table is None:
            block_table = getattr(metadata, "block_table", None)
        if block_table is None or block_table.ndim != 2:
            raise RuntimeError("target attention read observation requires a 2-D block table")
        block_table = block_table.detach()
        if block_table.shape[0] != batch_size:
            if block_table.shape[0] % batch_size != 0:
                raise RuntimeError("target attention read block table cannot be normalized to requests")
            verify_width = block_table.shape[0] // batch_size
            table_width = block_table.shape[1]
            if self._step_major_layout:
                rows = block_table.view(verify_width, batch_size, table_width).transpose(0, 1)
            else:
                rows = block_table.view(batch_size, verify_width, table_width)
            first_rows = rows[:, :1, :]
            if not torch.equal(rows, first_rows.expand_as(rows)):
                raise RuntimeError("target attention read block table changes within a request")
            block_table = rows[:, 0, :].contiguous()
        page_size = getattr(backend, "logical_page_size", None)
        if page_size is None:
            page_size = getattr(backend, "page_size", None)
        if page_size is None or int(page_size) <= 0:
            raise RuntimeError("target attention read observation requires a positive page size")
        kv_seq_lens = getattr(backend, "_actual_seq_kv", None)
        if _metadata_values_empty(kv_seq_lens):
            kv_seq_lens = getattr(backend, "_mla_actual_seq_kv_host", None)
        if _metadata_values_empty(kv_seq_lens):
            kv_seq_lens = getattr(metadata, "kv_seq_lens_host_values", None)
        if _metadata_values_empty(kv_seq_lens):
            kv_seq_lens = getattr(metadata, "kv_seq_lens", None)
        if _metadata_values_empty(kv_seq_lens):
            raise RuntimeError("target attention read observation requires KV sequence lengths")
        if isinstance(kv_seq_lens, torch.Tensor):
            kv_seq_lens = kv_seq_lens.detach().reshape(-1)
        else:
            kv_seq_lens = torch.as_tensor(kv_seq_lens, dtype=torch.long)
        if kv_seq_lens.numel() != batch_size:
            if kv_seq_lens.numel() % batch_size != 0:
                raise RuntimeError("target attention read KV lengths cannot be normalized to requests")
            verify_width = kv_seq_lens.numel() // batch_size
            rows = _normalize_request_major_rows(
                kv_seq_lens,
                batch_size,
                verify_width,
                step_major_layout=self._step_major_layout,
            )
            kv_seq_lens = rows[:, 0].contiguous()
        return block_table, kv_seq_lens, int(page_size)

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
            # Keep outputs and graph-visible metadata private to each step,
            # but share the role-owned temporary attention workspace. Draft
            # steps execute serially on the capture stream, so a workspace is
            # dead before the next step consumes the same slice.
            self._shared_execution_buffers = {}
            self._step_execution_states = tuple(
                AclGraphExecutionState({}, shared_persistent_buffers=self._shared_execution_buffers)
                for _ in self._metadata_by_step
            )
        elif len(self._step_execution_states) != len(self._metadata_by_step):
            raise RuntimeError("MTP role execution-state plan changed after warmup")

    def finish_warmup(self) -> None:
        """Mark fixed role backends ready for graph capture and replay."""
        if self._attention_backends:
            self._graph_prepared = True

    def require_direct_metadata_update(self) -> None:
        """Check once that every captured consumer borrows the owned tensors."""
        if not self._graph_prepared or not self._attention_backends:
            raise RuntimeError("sparse MTP binding requires prepared attention backends")
        if not all(getattr(backend, "graph_metadata_updated_in_place", False) for backend in self._attention_backends):
            raise RuntimeError("sparse MTP binding requires in-place graph metadata consumers")

    def update_repair_token_ids(self, repair_token_ids: torch.Tensor | None) -> None:
        if (self._repair_token_ids is None) != (repair_token_ids is None):
            raise RuntimeError("MTP repair-token presence changed after capture")
        if self._repair_token_ids is not None:
            assert repair_token_ids is not None
            if repair_token_ids.shape != self._repair_token_ids.shape:
                raise RuntimeError("MTP repair-token shape changed after capture")
            self._repair_token_ids.copy_(repair_token_ids)

    @torch.inference_mode()
    def update_metadata(
        self,
        metadata_by_step: Sequence[AttentionMetadata],
        repair_token_ids: torch.Tensor | None,
    ) -> None:
        """Copy a new invocation into the captured role's stable metadata."""
        if not self._graph_prepared:
            raise RuntimeError("MTP role metadata cannot update before graph warmup")
        if len(metadata_by_step) != len(self._metadata_by_step):
            raise RuntimeError("MTP role metadata step count changed")
        self.update_repair_token_ids(repair_token_ids)
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
            call_positions = (
                self._repair_positions
                if self._repair_positions is not None
                else torch.stack((positions - 1, positions), dim=1).reshape(-1)
            )
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
    target_probs: torch.Tensor | None = None
    committed_log_probs: torch.Tensor | None = None
    target_top_log_probs: torch.Tensor | None = None
    target_top_tokens: torch.Tensor | None = None
    # Optional diagnostic tensors. ``draft_hidden`` is also the recurrent
    # embedding passed into the following draft step.
    draft_hidden: torch.Tensor | None = None
    draft_logits: torch.Tensor | None = None
    draft_topk_indices: torch.Tensor | None = None
    target_logits: torch.Tensor | None = None
    target_topk_indices: torch.Tensor | None = None


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

    # The prefix length is the rejection index, or K for the bonus row.
    next_tokens = target_tokens.gather(1, accepted_count.to(torch.long).unsqueeze(1)).squeeze(1)
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
) -> torch.Tensor:
    """Materialize the runtime's fixed-width accepted/replacement layout."""

    committed = torch.cat((accepted_ids, torch.full_like(next_tokens.unsqueeze(1), -1)), dim=1)
    committed.scatter_(1, accepted_count.to(torch.long).unsqueeze(1), next_tokens.unsqueeze(1))
    return committed


def _runtime_token_views(token_state: torch.Tensor, batch: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_bytes = batch * (steps + 1) * 8
    tokens = token_state[:token_bytes].view(torch.int64).view(batch, steps + 1)
    counts = token_state[token_bytes:].view(torch.int32)
    return tokens, counts


def _compact_target_hidden(hidden: torch.Tensor, counts: torch.Tensor, steps: int) -> torch.Tensor:
    rows = torch.arange(counts.numel(), device=counts.device) * (steps + 1)
    current = rows + counts.to(torch.long)
    previous = rows + (counts.to(torch.long) - 1).clamp_min(0)
    return hidden.index_select(0, torch.stack((previous, current), dim=1).flatten())


def _greedy_runtime_commit(
    draft: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if draft.device.type == "npu":
        return torch.ops.xllm_ops.mtp_greedy_commit(draft, target, hidden)
    # The explicit eager/CPU oracle keeps the same storage contract. Online
    # NPU execution always uses the fused operator; errors propagate.
    accepted_ids, _, counts, next_tokens = greedy_acceptance(draft, target)
    tokens = _committed_tokens(accepted_ids, counts, next_tokens)
    return (
        torch.cat((tokens.flatten().view(torch.uint8), counts.view(torch.uint8))),
        _compact_target_hidden(hidden, counts, draft.shape[1]),
    )


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
        draft_sampling: MtpSamplingPlan | None = None,
        target_sampling: MtpSamplingPlan | None = None,
        sampling_random_inputs: MtpSamplingRandomInputs | None = None,
        trace_intermediates: bool | None = None,
        draft_greedy: LogitsFn | None = None,
        target_greedy: LogitsFn | None = None,
        runtime_outputs_only: bool = False,
        position_storage: MtpSparsePositionStorage | None = None,
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
        self.draft_greedy = draft_greedy
        self.target_greedy = target_greedy
        self.draft_activate = draft_activate
        self.target_activate = target_activate
        self.trace_intermediates = (
            os.environ.get("XLLM_MTP_STATE_ORACLE", "0") == "1" if trace_intermediates is None else trace_intermediates
        )
        self.runtime_outputs_only = runtime_outputs_only and not self.trace_intermediates
        self._position_storage = position_storage
        if draft_sampling is not None and draft_sampling.batch_size != batch_size:
            raise ValueError("draft sampling plan batch does not match MTP graph batch")
        if target_sampling is not None and target_sampling.batch_size != batch_size:
            raise ValueError("target sampling plan batch does not match MTP graph batch")
        if (draft_sampling is None) != (target_sampling is None):
            raise ValueError("draft and target sampling plans must be provided together")
        self.draft_sampling = None if draft_sampling is None else draft_sampling.clone_for_graph()
        self.target_sampling = None if target_sampling is None else target_sampling.clone_for_graph()
        self.sampling_random_inputs = (
            None if sampling_random_inputs is None else sampling_random_inputs.clone_for_graph()
        )
        self._validate_sampling_random_inputs()
        self.register_buffer(
            "_kv_seq_lens_template",
            kv_seq_lens.to(device=device, dtype=torch.int32).contiguous()
            if kv_seq_lens is not None
            else torch.empty(0, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_position_arena",
            position_storage.positions
            if position_storage is not None
            else torch.empty((2 * speculative_tokens + 1) * batch_size, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_target_position_offsets",
            torch.arange(speculative_tokens + 1, dtype=torch.long, device=device).unsqueeze(0)
            if position_storage is None
            else torch.empty(0, dtype=torch.long, device=device),
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

    def _validate_sampling_random_inputs(self) -> None:
        self._validate_sampling_random_inputs_for(self.sampling_random_inputs)

    def _validate_sampling_random_inputs_for(self, random_inputs: MtpSamplingRandomInputs | None) -> None:
        if random_inputs is None:
            return
        batch_size = self.batch_size
        steps = self.speculative_tokens
        vocab_size = self.vocab_size
        expected = {
            "draft_uniform": (batch_size, steps, vocab_size),
            "target_uniform": (batch_size, steps + 1, vocab_size),
            "acceptance_uniform": (batch_size, steps),
            "recovery_uniform": (batch_size, steps, vocab_size),
        }
        for name, shape in expected.items():
            value = getattr(random_inputs, name)
            if value is None:
                continue
            if value.device != self.device or value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32 on {self.device}")
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")

    def can_update_sampling_random_inputs(self, random_inputs: MtpSamplingRandomInputs | None) -> bool:
        if (self.sampling_random_inputs is None) != (random_inputs is None):
            return False
        if self.sampling_random_inputs is None:
            return True
        assert random_inputs is not None
        return self.sampling_random_inputs.layout_signature() == random_inputs.layout_signature()

    def update_sampling_random_inputs(self, random_inputs: MtpSamplingRandomInputs | None) -> None:
        if not self.can_update_sampling_random_inputs(random_inputs):
            raise RuntimeError("MTP random-input layout is incompatible with captured graph")
        if self.sampling_random_inputs is not None:
            assert random_inputs is not None
            self.sampling_random_inputs.update_from(random_inputs)

    def can_update_sampling_plans(
        self,
        draft_sampling: MtpSamplingPlan | None,
        target_sampling: MtpSamplingPlan | None,
    ) -> bool:
        draft_sampling = coerce_sampling_plan(draft_sampling, batch_size=self.batch_size)
        target_sampling = coerce_sampling_plan(target_sampling, batch_size=self.batch_size)
        if (self.draft_sampling is None) != (draft_sampling is None):
            return False
        if (self.target_sampling is None) != (target_sampling is None):
            return False
        if self.draft_sampling is None or self.target_sampling is None:
            return True
        assert draft_sampling is not None and target_sampling is not None
        return (
            self.draft_sampling.layout_signature() == draft_sampling.layout_signature()
            and self.target_sampling.layout_signature() == target_sampling.layout_signature()
        )

    def update_sampling_plans(
        self,
        draft_sampling: MtpSamplingPlan | None,
        target_sampling: MtpSamplingPlan | None,
    ) -> None:
        if not self.can_update_sampling_plans(draft_sampling, target_sampling):
            raise RuntimeError("MTP sampling plan is incompatible with captured graph")
        if self.draft_sampling is not None:
            assert draft_sampling is not None and self.target_sampling is not None and target_sampling is not None
            self.draft_sampling.update_from(draft_sampling)
            self.target_sampling.update_from(target_sampling)

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
    ) -> MtpGraphOutput | SpeculativeRuntimeOutput:
        if seed_token_ids.shape != (self.batch_size,):
            raise ValueError("seed_token_ids shape does not match the captured batch")
        if base_positions.shape != (self.batch_size,):
            raise ValueError("base_positions shape does not match the captured batch")
        if kv_seq_lens is not None and kv_seq_lens.shape != (self.batch_size,):
            raise ValueError("kv_seq_lens shape does not match the captured batch")

        draft_tokens: list[torch.Tensor] = []
        draft_probs: list[torch.Tensor] = []
        draft_hidden_rows: list[torch.Tensor] = []
        draft_logits_rows: list[torch.Tensor] = []
        draft_topk_rows: list[torch.Tensor | None] = []
        current_ids = seed_token_ids
        current_embedding = draft_input_embedding
        current_topk_indices = draft_topk_indices
        needs_acceptance_probs = self.target_sampling is not None and not self.target_sampling.all_greedy_sample
        use_draft_greedy = (
            self.draft_greedy is not None
            and not self.trace_intermediates
            and not needs_acceptance_probs
            and (self.draft_sampling is None or self.draft_sampling.plain_greedy)
        )
        for step in range(self.speculative_tokens):
            if self.draft_activate is not None:
                self.draft_activate()
            step_positions = self._position_arena[step * self.batch_size : (step + 1) * self.batch_size]
            if self._position_storage is None:
                torch.add(base_positions, step, out=step_positions)
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
            step_logits = None if use_draft_greedy else self.draft_logits(draft_hidden)
            if use_draft_greedy:
                current_ids = self.draft_greedy(draft_hidden)
            elif step_logits.shape != (self.batch_size, self.vocab_size):
                raise ValueError("draft logits must have shape [batch, vocab]")
            elif self.draft_sampling is None or (self.draft_sampling.plain_greedy and not needs_acceptance_probs):
                # Draft distributions and logprobs have no output consumer;
                # only probability acceptance needs them as intermediates.
                current_ids = step_logits.argmax(dim=-1)
            else:
                draft_uniform = None
                if self.sampling_random_inputs is not None:
                    draft_uniform = self.sampling_random_inputs.draft_uniform
                    if draft_uniform is not None:
                        draft_uniform = draft_uniform[:, step, :]
                sampled = sample_logits(
                    step_logits,
                    self.draft_sampling,
                    uniform=draft_uniform,
                    require_probs=needs_acceptance_probs,
                )
                current_ids = sampled.tokens
                if needs_acceptance_probs and sampled.probs is not None and self.draft_sampling.all_greedy_sample:
                    draft_probs.append(torch.zeros_like(sampled.probs).scatter_(-1, current_ids.unsqueeze(-1), 1.0))
                elif needs_acceptance_probs and sampled.probs is not None:
                    draft_probs.append(sampled.probs)
                del sampled
            if self.trace_intermediates:
                draft_hidden_rows.append(draft_hidden)
                draft_logits_rows.append(step_logits)
                draft_topk_rows.append(draft_output.topk_indices)
            # GLM MTP consumes the preceding body hidden as the next-step
            # embedding and optionally reuses the preceding DSA top-k state.
            current_embedding = draft_hidden
            current_topk_indices = draft_output.topk_indices
            draft_tokens.append(current_ids)
            # Release non-traced logits before the next body allocates its
            # workspace. The capture pool can reuse storage once it is dead.
            del step_logits, draft_hidden, draft_output
        del current_embedding, current_topk_indices
        draft_matrix = torch.stack(draft_tokens, dim=1)

        target_token_matrix = torch.cat((seed_token_ids.unsqueeze(1), draft_matrix), dim=1)
        target_ids = target_token_matrix.reshape(-1)
        target_positions = self._position_arena[self.speculative_tokens * self.batch_size :]
        if self._position_storage is None:
            torch.add(
                base_positions.unsqueeze(1),
                self._target_position_offsets,
                out=target_positions.view(self.batch_size, self.speculative_tokens + 1),
            )
        if self.target_activate is not None:
            self.target_activate()
        try:
            target_output = _body_output(self.target_forward(target_ids, target_positions, -1, None, None))
        except Exception as exc:
            raise RuntimeError(f"MTP target body failed: {exc}") from exc
        target_hidden = target_output.hidden
        if target_hidden.ndim != 2 or target_hidden.shape[0] != self.batch_size * (self.speculative_tokens + 1):
            raise ValueError("target hidden must have one row per target verification token")
        use_target_greedy = (
            self.target_greedy is not None
            and not self.trace_intermediates
            and (
                self.target_sampling is None
                or (
                    self.target_sampling.plain_greedy
                    and not self.target_sampling.return_probs
                    and not self.target_sampling.logprobs
                    and self.target_sampling.max_top_logprobs == 0
                )
            )
        )
        target_logits = None if use_target_greedy else self.target_logits(target_hidden)
        if target_logits is not None and target_logits.shape != (
            self.batch_size * (self.speculative_tokens + 1),
            self.vocab_size,
        ):
            raise ValueError("target logits must have shape [batch*(K+1), vocab]")
        target_probs = None
        target_log_probs = None
        target_log_normalizer = None
        plain_greedy = self.target_sampling is not None and self.target_sampling.plain_greedy
        if use_target_greedy:
            target_tokens = self.target_greedy(target_hidden).reshape(self.batch_size, self.speculative_tokens + 1)
        elif self.target_sampling is None or (plain_greedy and not self.target_sampling.return_probs):
            target_tokens = target_logits.argmax(dim=-1).reshape(self.batch_size, self.speculative_tokens + 1)
            if self.target_sampling is not None and (
                self.target_sampling.logprobs or self.target_sampling.max_top_logprobs > 0
            ):
                # Only committed and top-k scores leave the graph. A scalar
                # normalizer per row avoids a full-vocabulary log-prob tensor.
                target_log_normalizer = torch.logsumexp(target_logits.float(), dim=-1).reshape(
                    self.batch_size, self.speculative_tokens + 1, 1
                )
        else:
            target_uniform = None
            if self.sampling_random_inputs is not None:
                target_uniform = self.sampling_random_inputs.target_uniform
                if target_uniform is not None:
                    target_uniform = target_uniform.reshape(
                        self.batch_size * (self.speculative_tokens + 1), self.vocab_size
                    )
            sampled_target = sample_logits(
                target_logits,
                self.target_sampling.expand_for_rows(self.batch_size * (self.speculative_tokens + 1)),
                uniform=target_uniform,
            )
            target_tokens = sampled_target.tokens.reshape(self.batch_size, self.speculative_tokens + 1)
            if sampled_target.probs is not None:
                target_probs = sampled_target.probs.reshape(
                    self.batch_size, self.speculative_tokens + 1, self.vocab_size
                )
            if sampled_target.log_probs is not None:
                target_log_probs = sampled_target.log_probs.reshape(
                    self.batch_size, self.speculative_tokens + 1, self.vocab_size
                )

        token_state = None
        compact_hidden = None
        greedy = self.target_sampling is None or self.target_sampling.all_greedy_sample
        if self.runtime_outputs_only and greedy:
            token_state, compact_hidden = _greedy_runtime_commit(draft_matrix, target_tokens, target_hidden)
            committed_tokens, accepted_count = _runtime_token_views(
                token_state, self.batch_size, self.speculative_tokens
            )
        elif greedy:
            (
                accepted_ids,
                accepted_mask,
                accepted_count,
                next_tokens,
            ) = greedy_acceptance(draft_matrix, target_tokens)
        else:
            assert target_probs is not None
            draft_prob_matrix = torch.stack(draft_probs, dim=1)
            acceptance_uniform = None
            recovery_uniform = None
            if self.sampling_random_inputs is not None:
                acceptance_uniform = self.sampling_random_inputs.acceptance_uniform
                recovery_uniform = self.sampling_random_inputs.recovery_uniform
            (
                accepted_ids,
                accepted_mask,
                accepted_count,
                next_tokens,
            ) = probabilistic_acceptance(
                draft_matrix,
                draft_prob_matrix,
                target_tokens,
                target_probs,
                self.target_sampling.request_do_sample(device=target_tokens.device),
                acceptance_uniform=acceptance_uniform,
                recovery_uniform=recovery_uniform,
            )
        if token_state is None:
            committed_tokens = _committed_tokens(
                accepted_ids,
                accepted_count,
                next_tokens,
            )
        committed_log_probs = None
        target_top_log_probs = None
        target_top_tokens = None
        score_rows = target_log_probs
        if target_log_normalizer is not None:
            score_rows = target_logits.view(self.batch_size, self.speculative_tokens + 1, self.vocab_size)
        if score_rows is not None and self.target_sampling is not None and self.target_sampling.logprobs:
            committed_indices = committed_tokens.clamp_min(0).unsqueeze(-1)
            selected_scores = score_rows.gather(-1, committed_indices).float()
            if target_log_normalizer is not None:
                selected_scores = selected_scores - target_log_normalizer
            committed_log_probs = selected_scores.squeeze(-1)
            committed_log_probs = torch.where(
                committed_tokens.ge(0), committed_log_probs, torch.zeros_like(committed_log_probs)
            )
        if score_rows is not None and self.target_sampling is not None and self.target_sampling.max_top_logprobs > 0:
            top_width = min(self.target_sampling.max_top_logprobs, self.vocab_size)
            target_top_log_probs, target_top_tokens = score_rows.topk(top_width, dim=-1)
            if target_log_normalizer is not None:
                target_top_log_probs = target_top_log_probs.float() - target_log_normalizer
        if self.runtime_outputs_only:
            if compact_hidden is None:
                compact_hidden = _compact_target_hidden(target_hidden, accepted_count, self.speculative_tokens)
            return SpeculativeRuntimeOutput(
                committed_tokens=committed_tokens,
                accepted_count=accepted_count,
                target_embeddings=compact_hidden,
                token_state=token_state,
                logprobs=committed_log_probs,
                top_logprobs=target_top_log_probs,
                top_tokens=target_top_tokens,
                target_probs=(
                    target_probs if self.target_sampling is not None and self.target_sampling.return_probs else None
                ),
            )
        draft_hidden_matrix = None
        draft_logits_matrix = None
        draft_topk_matrix = None
        target_logits_matrix = None
        target_topk_trace_matrix = None
        if self.trace_intermediates:
            draft_hidden_matrix = torch.stack(draft_hidden_rows, dim=1)
            draft_logits_matrix = torch.stack(draft_logits_rows, dim=1)
            if any(value is not None for value in draft_topk_rows):
                if any(value is None for value in draft_topk_rows):
                    raise RuntimeError("MTP draft top-k state changed presence across fixed steps")
                draft_topk_matrix = torch.stack([value for value in draft_topk_rows if value is not None], dim=1)
            target_logits_matrix = target_logits.reshape(
                self.batch_size,
                self.speculative_tokens + 1,
                self.vocab_size,
            )
            if target_output.topk_indices is not None:
                target_topk = target_output.topk_indices
                if target_topk.shape[0] != self.batch_size * (self.speculative_tokens + 1):
                    raise ValueError("target top-k state must have one row per target verification token")
                target_topk_trace_matrix = target_topk.reshape(
                    self.batch_size,
                    self.speculative_tokens + 1,
                    *target_topk.shape[1:],
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
            target_probs=(
                target_probs if self.target_sampling is not None and self.target_sampling.return_probs else None
            ),
            committed_log_probs=committed_log_probs,
            target_top_log_probs=target_top_log_probs,
            target_top_tokens=target_top_tokens,
            draft_hidden=draft_hidden_matrix,
            draft_logits=draft_logits_matrix,
            draft_topk_indices=draft_topk_matrix,
            target_logits=target_logits_matrix,
            target_topk_indices=target_topk_trace_matrix,
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
        kv_payload_oracle: MtpKvPayloadOracle | None = None,
    ) -> None:
        if backend not in ("eager", "aclgraph"):
            raise ValueError(f"unknown MTP graph backend: {backend!r}")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        self.recipe = recipe
        self.backend = backend
        self.warmup_steps = warmup_steps
        self.prepare = prepare
        self.kv_payload_oracle = kv_payload_oracle
        self._closed = False
        self._captured = False
        self._graph = None
        self._capture_stream = None
        self._graph_tasks: list[AclGraphTask] = []
        self._execution_states = (
            (AclGraphExecutionState({}), AclGraphExecutionState({})) if backend == "aclgraph" else (None, None)
        )
        self._entry = MtpGraphEntry(
            MtpGraphCapability(
                speculative_tokens=recipe.speculative_tokens,
                batch_size=recipe.batch_size,
                vocab_size=recipe.vocab_size,
                sampling_mode=(recipe.target_sampling.mode if recipe.target_sampling is not None else "greedy"),
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
        self._static_output: MtpGraphOutput | SpeculativeRuntimeOutput | None = None

    def _prepare_static_inputs(self) -> None:
        if self.prepare is not None:
            self.prepare(
                self._static_seed_token_ids,
                self._static_base_positions,
                self._static_kv_seq_lens,
            )

    def close(self) -> None:
        """Drain replay and output copies before releasing graph storage."""
        if self._closed:
            return
        if self._capture_stream is not None:
            self._capture_stream.synchronize()
            # Output snapshots are enqueued on the caller stream after it
            # waits for replay. They must finish before graph.reset frees it.
            torch.npu.current_stream(self.recipe.device).synchronize()
        if self._graph is not None:
            self._graph.reset()
        self._graph = None
        self._entry.graph = None
        self._entry.captured = False
        self._captured = False
        self._static_output = None
        self._graph_tasks.clear()
        self.recipe.bind_graph_contexts(None, None, None)
        self._execution_states = (None, None)
        self._capture_stream = None
        self._closed = True

    @property
    def capability(self) -> MtpGraphCapability:
        return self._entry.capability

    @property
    def entry(self) -> MtpGraphEntry:
        return self._entry

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def draft_embedding_destination(self) -> torch.Tensor | None:
        """Borrow the final input buffer; producer must follow replay's read event."""
        return self._static_draft_input_embedding

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
        token_dtypes = (torch.int32, torch.int64) if self.backend == "aclgraph" else (torch.int64,)
        if seed_token_ids.dtype not in token_dtypes or base_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("MTP graph inputs require integer ids/positions; eager token ids must be int64")
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

    def _run_static(self) -> MtpGraphOutput | SpeculativeRuntimeOutput:
        return self.recipe(
            self._static_seed_token_ids,
            self._static_base_positions,
            self._static_kv_seq_lens,
            self._static_draft_input_embedding,
            self._static_draft_topk_indices,
        )

    @staticmethod
    def _clone_graph_output(
        output: MtpGraphOutput | SpeculativeRuntimeOutput,
    ) -> MtpGraphOutput | SpeculativeRuntimeOutput:
        """Detach one replay result from the persistent ACL graph buffers.

        ACL graph replay writes the same output allocations on every call.
        Schedule overlap can keep an earlier MTP result alive while the next
        draft or target invocation starts, so returning those allocations
        directly allows a later replay to overwrite data still in use.
        """

        def clone(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.clone()

        if isinstance(output, SpeculativeRuntimeOutput):
            token_state = clone(output.token_state)
            if token_state is None:
                tokens, counts = output.committed_tokens.clone(), output.accepted_count.clone()
            else:
                batch, width = output.committed_tokens.shape
                tokens, counts = _runtime_token_views(token_state, batch, width - 1)
            return SpeculativeRuntimeOutput(
                committed_tokens=tokens,
                accepted_count=counts,
                target_embeddings=output.target_embeddings.clone(),
                token_state=token_state,
                logprobs=clone(output.logprobs),
                top_logprobs=clone(output.top_logprobs),
                top_tokens=clone(output.top_tokens),
                target_probs=clone(output.target_probs),
            )
        return MtpGraphOutput(
            accepted_ids=output.accepted_ids.clone(),
            accepted_mask=output.accepted_mask.clone(),
            accepted_count=output.accepted_count.clone(),
            next_tokens=output.next_tokens.clone(),
            committed_tokens=output.committed_tokens.clone(),
            draft_tokens=output.draft_tokens.clone(),
            target_tokens=output.target_tokens.clone(),
            next_positions=output.next_positions.clone(),
            next_kv_seq_lens=clone(output.next_kv_seq_lens),
            next_embeddings=clone(output.next_embeddings),
            next_topk_indices=clone(output.next_topk_indices),
            target_embeddings=output.target_embeddings.clone(),
            target_probs=clone(output.target_probs),
            committed_log_probs=clone(output.committed_log_probs),
            target_top_log_probs=clone(output.target_top_log_probs),
            target_top_tokens=clone(output.target_top_tokens),
            draft_hidden=clone(output.draft_hidden),
            draft_logits=clone(output.draft_logits),
            draft_topk_indices=clone(output.draft_topk_indices),
            target_logits=clone(output.target_logits),
            target_topk_indices=clone(output.target_topk_indices),
        )

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

    def update_sampling_plans(
        self,
        draft_sampling: MtpSamplingPlan | None,
        target_sampling: MtpSamplingPlan | None,
    ) -> None:
        """Copy request sampling controls into graph-owned stable buffers."""
        if not self._captured:
            raise RuntimeError("MTP ACL graph must be captured before sampling update")
        draft_sampling = coerce_sampling_plan(draft_sampling, batch_size=self.recipe.batch_size)
        target_sampling = coerce_sampling_plan(target_sampling, batch_size=self.recipe.batch_size)
        self.recipe.update_sampling_plans(draft_sampling, target_sampling)

    def can_update_sampling_random_inputs(self, random_inputs: MtpSamplingRandomInputs | None) -> bool:
        """Check whether fixed random controls fit the captured graph."""
        if not self._captured:
            return False
        if random_inputs is not None:
            self.recipe._validate_sampling_random_inputs_for(random_inputs)
        return self.recipe.can_update_sampling_random_inputs(random_inputs)

    def update_sampling_random_inputs(self, random_inputs: MtpSamplingRandomInputs | None) -> None:
        """Update graph-owned fixed random inputs without changing its layout."""
        if not self._captured:
            raise RuntimeError("MTP ACL graph must be captured before random-input update")
        self.recipe._validate_sampling_random_inputs_for(random_inputs)
        self.recipe.update_sampling_random_inputs(random_inputs)

    def capture(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None = None,
        draft_input_embedding: torch.Tensor | None = None,
        draft_topk_indices: torch.Tensor | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("MTP graph runner is closed")
        self.capability.require_aclgraph()
        self._validate_inputs(seed_token_ids, base_positions, kv_seq_lens)
        if draft_input_embedding is not None:
            self._static_draft_input_embedding = torch.empty_like(draft_input_embedding)
        if draft_topk_indices is not None:
            self._static_draft_topk_indices = torch.empty_like(draft_topk_indices)
        self._validate_recurrent_inputs(draft_input_embedding, draft_topk_indices)
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            raise RuntimeError("ACL graph backend requires torch.npu.NPUGraph")
        initial_cache = self.kv_payload_oracle.snapshot() if self.kv_payload_oracle is not None else None
        stage = "input copy"
        capture_error: RuntimeError | None = None
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
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                stage = "warmup collective barrier"
                torch.distributed.barrier()

            stage = "graph capture"
            self._graph = torch.npu.NPUGraph()
            capture_context = AclGraphCaptureContext(self._capture_stream, [])
            self.recipe.bind_graph_contexts(capture_context, *self._execution_states)
            with torch.npu.stream(self._capture_stream), torch.npu.graph(self._graph, stream=self._capture_stream):
                self._static_output = self._run_static()
            self._graph_tasks = capture_context.tasks
        except Exception as exc:
            capture_error = RuntimeError(f"MTP ACL graph capture failed during {stage}: {exc}")
        finally:
            if initial_cache is not None:
                try:
                    if self._capture_stream is not None:
                        self._capture_stream.synchronize()
                    initial_cache.restore()
                    torch.npu.current_stream().synchronize()
                except Exception as restore_error:
                    if capture_error is None:
                        raise RuntimeError("MTP ACL graph capture cache restore failed") from restore_error
                    raise RuntimeError(f"{capture_error}; cache restore failed: {restore_error}") from capture_error
        if capture_error is not None:
            raise capture_error
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
    ) -> SpeculativeExecutionOutput | SpeculativeRuntimeOutput:
        if self._closed:
            raise RuntimeError("MTP graph runner is closed")
        self._validate_inputs(seed_token_ids, base_positions, kv_seq_lens)
        self._validate_recurrent_inputs(draft_input_embedding, draft_topk_indices)
        reference = None
        if self.kv_payload_oracle is not None:
            self.kv_payload_oracle.observe_next_attention_read_set(base_positions)
            reference = self.kv_payload_oracle.run_reference(
                seed_token_ids.to(torch.long), base_positions, kv_seq_lens, draft_input_embedding, draft_topk_indices
            )
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
                if (
                    draft_input_embedding.data_ptr() != self._static_draft_input_embedding.data_ptr()
                    or draft_input_embedding.stride() != self._static_draft_input_embedding.stride()
                ):
                    self._static_draft_input_embedding.copy_(draft_input_embedding)
            if self._static_draft_topk_indices is not None:
                assert draft_topk_indices is not None
                self._static_draft_topk_indices.copy_(draft_topk_indices)
            self._prepare_static_inputs()
            self._capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(self._capture_stream):
                self._update_graph_tasks(self._capture_stream)
                self._graph.replay()
            torch.npu.current_stream().wait_stream(self._capture_stream)
            assert self._static_output is not None
            output = self._clone_graph_output(self._static_output)

        if isinstance(output, SpeculativeRuntimeOutput):
            return output
        next_state = SpeculativeDeviceState(
            token_ids=output.next_tokens,
            positions=output.next_positions,
            kv_seq_lens=output.next_kv_seq_lens,
            embeddings=output.next_embeddings,
            cache_slots=None,
            topk_indices=output.next_topk_indices,
            committed_mask=output.accepted_mask,
        )
        result = SpeculativeExecutionOutput(
            accepted_ids=output.accepted_ids,
            accepted_mask=output.accepted_mask,
            accepted_count=output.accepted_count,
            next_state=next_state,
            committed_tokens=output.committed_tokens,
            target_embeddings=output.target_embeddings,
            draft_tokens=output.draft_tokens,
            target_tokens=output.target_tokens,
            logprobs=output.committed_log_probs,
            top_logprobs=output.target_top_log_probs,
            top_tokens=output.target_top_tokens,
            target_probs=output.target_probs,
            draft_hidden=output.draft_hidden,
            draft_logits=output.draft_logits,
            draft_topk_indices=output.draft_topk_indices,
            target_logits=output.target_logits,
            target_topk_indices=output.target_topk_indices,
        )
        if self.kv_payload_oracle is not None:
            assert reference is not None
            self.kv_payload_oracle.compare(result, reference)
            self.kv_payload_oracle.record_rejected_target_slots(result)
        return result

    def _update_graph_tasks(self, stream: object) -> None:
        """Refresh captured attention arguments before serial replay."""
        for task in self._graph_tasks:
            torch.npu.graph_task_update_begin(stream, task.handle)
            task.update()
            torch.npu.graph_task_update_end(stream)
            task.event.record(stream)


class MtpGraphVariantRegistry:
    """Own captured MTP recipes and their FIFO lifetime in Python.

    The C++ worker only supplies metadata, input tensors, and sampling
    controls.  Compatibility checks, capture, replay updates, and bounded
    eviction stay with the Python objects that own the recipe and its graph
    addresses.  Entries are intentionally FIFO: reusing an entry does not
    change its eviction order, which keeps graph lifetime deterministic.
    """

    def __init__(
        self,
        target_executor: object,
        draft_executor: object,
        *,
        max_variants: int = 8,
        draft_activate: ActivateFn | None = None,
        target_activate: ActivateFn | None = None,
        capture_runner: Callable[..., None] | None = None,
        draft_metadata_factory: SparseMetadataFactory | None = None,
        target_metadata_factory: SparseMetadataFactory | None = None,
    ) -> None:
        if max_variants <= 0:
            raise ValueError("MTP graph variant capacity must be positive")
        if not hasattr(target_executor, "create_mtp_graph_runner_from_metadata"):
            raise TypeError("target executor cannot create MTP graph runners")
        self._target_executor = target_executor
        self._draft_executor = draft_executor
        self._max_variants = max_variants
        self._draft_activate = draft_activate
        self._target_activate = target_activate
        self._capture_runner = capture_runner
        self._draft_metadata_factory = draft_metadata_factory
        self._target_metadata_factory = target_metadata_factory
        self._variants: dict[tuple[object, ...], MtpAclGraphRunner] = {}
        self._sparse_bindings: dict[tuple[object, ...], MtpSparseMetadataBinding] = {}
        self.draft_embedding_destination: torch.Tensor | None = None

    @property
    def variant_count(self) -> int:
        return len(self._variants)

    def _evict_if_full(self) -> None:
        if len(self._variants) >= self._max_variants:
            logger.warning(
                "MTP Python graph variant capacity reached; evicting oldest: capacity=%d", self._max_variants
            )
            oldest_key = next(iter(self._variants))
            self._retire_variant(oldest_key)

    def _retire_variant(self, key: tuple[object, ...]) -> None:
        self._variants[key].close()
        del self._variants[key]
        self._sparse_bindings.pop(key, None)

    @torch.inference_mode()
    def execute_sparse(
        self,
        block_table: torch.Tensor,
        first_kv_seq_lens: torch.Tensor,
        first_slots: torch.Tensor,
        repair_token_ids: torch.Tensor,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        draft_input_embedding: torch.Tensor,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        block_size: int,
        target_step_major_layout: bool = False,
        return_probs: bool = False,
        logprobs: bool = False,
        max_top_logprobs: int = 0,
    ) -> SpeculativeExecutionOutput | SpeculativeRuntimeOutput:
        """Execute the admitted sparse/greedy path using final graph storage.

        Sampling flags are immutable for a variant. The Worker has already
        rejected per-request distribution transforms before this entry point.
        """
        capacity = 1 << (max(32, block_table.shape[1]) - 1).bit_length()
        key = (
            "sparse",
            batch_size,
            speculative_tokens,
            vocab_size,
            block_size,
            target_step_major_layout,
            # Token/position destinations have fixed integer dtypes; their
            # caller dtype/stride does not change the captured graph. Reuse
            # the same variant across bootstrap and fused continuation.
            draft_input_embedding.shape,
            draft_input_embedding.dtype,
            draft_input_embedding.device,
            return_probs,
            logprobs,
            max_top_logprobs,
        )
        variant = self._variants.get(key)
        if variant is not None and self._sparse_bindings[key].table_capacity < capacity:
            # A single capacity per batch/output contract: grow before capture,
            # reuse it on shrink, and clear unused columns in the fused update.
            # Retirement waits for replay/snapshot consumers before allocation.
            self._retire_variant(key)
            variant = None
        if variant is None:
            self._evict_if_full()
            draft_factory = self._draft_metadata_factory
            target_factory = self._target_metadata_factory
            if draft_factory is None or target_factory is None:
                raise RuntimeError("sparse MTP registry requires native metadata factories")
            draft_plan = MtpSamplingPlan(batch_size=batch_size, return_probs=False)
            target_plan = MtpSamplingPlan(
                batch_size=batch_size, return_probs=return_probs, logprobs=logprobs, max_top_logprobs=max_top_logprobs
            )

            draft_storage = MtpSparseMetadataStorage(
                draft_factory,
                [batch_size * (2 if step == 0 else 1) for step in range(speculative_tokens)],
                capacity,
                block_table.device,
                share_block_tables=True,
            )
            target_storage = MtpSparseMetadataStorage(
                target_factory, [batch_size * (speculative_tokens + 1)], capacity, block_table.device
            )
            position_storage = MtpSparsePositionStorage(batch_size, speculative_tokens, block_table.device)
            variant = self._target_executor.create_mtp_graph_runner_from_metadata(
                self._draft_executor,
                draft_storage.metadata,
                target_storage.metadata[0],
                repair_token_ids=repair_token_ids,
                batch_size=batch_size,
                speculative_tokens=speculative_tokens,
                vocab_size=vocab_size,
                kv_seq_lens=kv_seq_lens,
                target_step_major_layout=target_step_major_layout,
                draft_activate=self._draft_activate,
                target_activate=self._target_activate,
                draft_sampling=draft_plan,
                target_sampling=target_plan,
                runtime_outputs_only=True,
                draft_metadata_storage=draft_storage,
                target_metadata_storage=target_storage,
                position_storage=position_storage,
            )
            draft_role = variant.recipe.draft_forward
            target_role = variant.recipe.target_forward
            binding = MtpSparseMetadataBinding(
                draft_role._metadata_by_step,
                draft_role._metadata_arenas,
                target_role._metadata_by_step,
                target_role._metadata_arenas,
                position_storage=position_storage,
                batch_size=batch_size,
                speculative_tokens=speculative_tokens,
                table_capacity=capacity,
                block_size=block_size,
                target_step_major=target_step_major_layout,
            )
            binding.update(block_table, base_positions, kv_seq_lens, first_kv_seq_lens, first_slots)
            logger.info("MTP Python graph variant capture: index=%d", len(self._variants))
            inputs = (seed_token_ids, base_positions, kv_seq_lens, draft_input_embedding, None)
            if self._capture_runner is None:
                variant.capture(*inputs)
            else:
                self._capture_runner(variant, *inputs)
            draft_role.require_direct_metadata_update()
            target_role.require_direct_metadata_update()
            self._variants[key] = variant
            self._sparse_bindings[key] = binding
        else:
            self._sparse_bindings[key].update(block_table, base_positions, kv_seq_lens, first_kv_seq_lens, first_slots)
            variant.recipe.draft_forward.update_repair_token_ids(repair_token_ids)
            logger.debug("MTP Python graph variant replay: count=%d", len(self._variants))
        output = variant.execute(seed_token_ids, base_positions, kv_seq_lens, draft_input_embedding)
        # Export the destination for the next C++ fused prepare. A capacity or
        # output-layout transition copies once into the newly resolved variant
        # and publishes its replacement destination after execution.
        self.draft_embedding_destination = variant.draft_embedding_destination
        return output

    def execute(
        self,
        draft_metadata: Sequence[object],
        target_metadata: object,
        *,
        repair_token_ids: torch.Tensor,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        draft_input_embedding: torch.Tensor,
        draft_topk_indices: torch.Tensor | None = None,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        target_step_major_layout: bool = False,
        draft_sampling: MtpSamplingPlan | dict[str, object] | None = None,
        target_sampling: MtpSamplingPlan | dict[str, object] | None = None,
    ) -> SpeculativeExecutionOutput | SpeculativeRuntimeOutput:
        draft_sampling_plan = coerce_sampling_plan(draft_sampling, batch_size=batch_size)
        target_sampling_plan = coerce_sampling_plan(target_sampling, batch_size=batch_size)
        metadata_by_step = tuple(draft_metadata)
        key = (
            batch_size,
            speculative_tokens,
            vocab_size,
            target_step_major_layout,
            tuple(_metadata_signature(metadata) for metadata in metadata_by_step),
            _metadata_signature(target_metadata),
            tuple(
                _tensor_signature(value)
                for value in (
                    repair_token_ids,
                    seed_token_ids,
                    base_positions,
                    kv_seq_lens,
                    draft_input_embedding,
                    draft_topk_indices,
                )
            ),
            None if draft_sampling_plan is None else draft_sampling_plan.layout_signature(),
            None if target_sampling_plan is None else target_sampling_plan.layout_signature(),
        )
        variant = self._variants.get(key)
        if variant is not None:
            # update_* owns validation. Preflighting again here repeats the
            # same metadata/signature walks on every hot replay.
            variant.update_metadata(metadata_by_step, target_metadata, repair_token_ids)
            variant.update_sampling_plans(draft_sampling_plan, target_sampling_plan)
            logger.debug("MTP Python graph variant replay: count=%d", len(self._variants))
            return variant.execute(
                seed_token_ids,
                base_positions,
                kv_seq_lens,
                draft_input_embedding,
                draft_topk_indices,
            )

        self._evict_if_full()
        logger.info("MTP Python graph variant capture: index=%d", len(self._variants))
        variant = self._target_executor.create_mtp_graph_runner_from_metadata(
            self._draft_executor,
            metadata_by_step,
            target_metadata,
            repair_token_ids=repair_token_ids,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=vocab_size,
            kv_seq_lens=kv_seq_lens,
            target_step_major_layout=target_step_major_layout,
            draft_activate=self._draft_activate,
            target_activate=self._target_activate,
            draft_sampling=draft_sampling_plan,
            target_sampling=target_sampling_plan,
            runtime_outputs_only=True,
        )
        capture = (
            variant.capture if self._capture_runner is None else lambda *args: self._capture_runner(variant, *args)
        )
        capture(
            seed_token_ids,
            base_positions,
            kv_seq_lens,
            draft_input_embedding,
            draft_topk_indices,
        )
        self._variants[key] = variant
        return variant.execute(
            seed_token_ids,
            base_positions,
            kv_seq_lens,
            draft_input_embedding,
            draft_topk_indices,
        )
