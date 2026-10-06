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

"""Write all sparse MTP role metadata directly into graph-owned arenas."""

import tilelang.language as T
from tilelang import tvm

MTP_SPARSE_METADATA_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_mtp_sparse_metadata_update_kernel(position_bits: int) -> tvm.tir.PrimFunc:
    if position_bits not in (32, 64):
        raise ValueError("MTP base positions must be int32 or int64")
    position_dtype = f"int{position_bits}"

    @T.prim_func
    def mtp_sparse_metadata_update(
        table_handle: T.handle,
        positions_handle: T.handle,
        lengths_handle: T.handle,
        first_lengths_handle: T.handle,
        first_slots_handle: T.handle,
        layout_handle: T.handle,
        draft_handle: T.handle,
        target_handle: T.handle,
        role_positions_handle: T.handle,
        batch_size: T.int32,
        speculative_tokens: T.int32,
        source_columns: T.int32,
        source_stride: T.int32,
        table_capacity: T.int32,
        block_size: T.int32,
        step_major: T.int32,
        draft_words: T.int32,
        target_words: T.int32,
    ):
        table = T.match_buffer(table_handle, ((batch_size - 1) * source_stride + source_columns,), "int32")
        positions = T.match_buffer(positions_handle, (T.max(batch_size, 1),), position_dtype)
        lengths = T.match_buffer(lengths_handle, (T.max(batch_size, 1),), "int32")
        first_lengths = T.match_buffer(first_lengths_handle, (batch_size * 2,), "int32")
        first_slots = T.match_buffer(first_slots_handle, (batch_size * 2,), "int32")
        # Per role: slot offset, KV-length offset, block-table offset, in words.
        layout = T.match_buffer(layout_handle, ((speculative_tokens + 1) * 3,), "int32")
        draft = T.match_buffer(draft_handle, (T.max(draft_words, 1),), "int32")
        target = T.match_buffer(target_handle, (T.max(target_words, 1),), "int32")
        role_positions = T.match_buffer(role_positions_handle, ((2 * speculative_tokens + 3) * batch_size,), "int64")
        with T.Kernel(1, is_npu=True) as (_, vid), T.Scope("V"):
            if vid == 0:
                # One writer also avoids sharing metadata cache lines between
                # cores for the small B=1/3/8 decode batches.
                table_tile = T.alloc_ub((32,), "int32")
                slot_tile = T.alloc_ub((32,), "int32")
                length_tile = T.alloc_ub((32,), "int32")
                position_tile = T.alloc_ub((32,), "int64")
                rows = T.alloc_var("int32")
                row = T.alloc_var("int32")
                step = T.alloc_var("int32")
                target_row = T.alloc_var("int32")
                repair_index = T.alloc_var("int32")
                # Body positions precede the target role's optional
                # request-major to step-major permutation.
                for chunk in T.serial(T.ceildiv((2 * speculative_tokens + 3) * batch_size, 32)):
                    count = T.min(32, (2 * speculative_tokens + 3) * batch_size - chunk * 32)
                    for lane in T.serial(count):
                        index = chunk * 32 + lane
                        if index < speculative_tokens * batch_size:
                            row = index % batch_size
                            step = index // batch_size
                        elif index < (2 * speculative_tokens + 1) * batch_size:
                            target_index = index - speculative_tokens * batch_size
                            row = target_index // (speculative_tokens + 1)
                            step = target_index % (speculative_tokens + 1)
                        else:
                            # Materialize the nonnegative index. Expanding
                            # (chunk*32 + lane - start)//2 can introduce a
                            # negative partial dividend, whose C++ truncation
                            # differs from floor division for odd batch sizes.
                            repair_index = index - (2 * speculative_tokens + 1) * batch_size
                            row = repair_index // 2
                            step = repair_index % 2 - 1
                        position_tile[lane] = T.Cast("int64", positions[row]) + T.Cast("int64", step)
                    T.copy(position_tile, role_positions[chunk * 32 : chunk * 32 + count])
                # Write scalar metadata via UB->GM copies. Scalar GM stores
                # would depend on device cache flushes and can be assigned to
                # the wrong core kind by automatic cube/vector partitioning.
                for role in T.serial(speculative_tokens + 1):
                    rows = batch_size
                    if role == 0:
                        rows = batch_size * 2
                    if role == speculative_tokens:
                        rows = batch_size * (speculative_tokens + 1)
                    for chunk in T.serial(T.ceildiv(rows, 32)):
                        count = T.min(32, rows - chunk * 32)
                        if role == 0:
                            T.copy(first_slots[chunk * 32 : chunk * 32 + count], slot_tile)
                            T.copy(first_lengths[chunk * 32 : chunk * 32 + count], length_tile)
                        else:
                            for lane in T.serial(count):
                                index = chunk * 32 + lane
                                row = index
                                step = role
                                if role == speculative_tokens:
                                    if step_major != 0:
                                        row = index % batch_size
                                        step = index // batch_size
                                    else:
                                        row = index // (speculative_tokens + 1)
                                        step = index % (speculative_tokens + 1)
                                position = T.Cast("int32", positions[row]) + step
                                column = position // block_size
                                physical_block = table[
                                    row * source_stride + T.min(T.max(column, 0), source_columns - 1)
                                ]
                                valid_block = T.Select((column >= 0) & (column < source_columns), physical_block, 0)
                                slot_tile[lane] = valid_block * block_size + position % block_size
                                length_tile[lane] = lengths[row] + step
                        slot_offset = layout[role * 3] + chunk * 32
                        length_offset = layout[role * 3 + 1] + chunk * 32
                        if role < speculative_tokens:
                            T.copy(slot_tile, draft[slot_offset : slot_offset + count])
                            T.copy(length_tile, draft[length_offset : length_offset + count])
                        else:
                            T.copy(slot_tile, target[slot_offset : slot_offset + count])
                            T.copy(length_tile, target[length_offset : length_offset + count])
                for source_row in T.serial(batch_size):
                    for chunk in T.serial(table_capacity // 32):
                        T.tile.fill(table_tile, 0)
                        column = chunk * 32
                        count = T.min(32, T.max(0, source_columns - column))
                        if count > 0:
                            start = source_row * source_stride + column
                            T.copy(table[start : start + count], table_tile)
                        # Every replay clears the unused table tail, including
                        # shrinking within a capacity bucket. No Host scan.
                        for repair_row in T.serial(2):
                            T.copy(
                                table_tile, draft[layout[2] + (source_row * 2 + repair_row) * table_capacity + column]
                            )
                        for step in T.serial(1, speculative_tokens):
                            # Same-layout draft tables share a destination.
                            # They contain identical pages, so write it once
                            # before replay; dynamic fields remain per step.
                            if step == 1 or layout[step * 3 + 2] != layout[(step - 1) * 3 + 2]:
                                T.copy(table_tile, draft[layout[step * 3 + 2] + source_row * table_capacity + column])
                        for step in T.serial(speculative_tokens + 1):
                            if step_major != 0:
                                target_row = step * batch_size + source_row
                            else:
                                target_row = source_row * (speculative_tokens + 1) + step
                            T.copy(
                                table_tile,
                                target[layout[speculative_tokens * 3 + 2] + target_row * table_capacity + column],
                            )

    return mtp_sparse_metadata_update
