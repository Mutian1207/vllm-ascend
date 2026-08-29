# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)

_QUERY_BLOCK_SIZE = 16


def _get_workers_per_req(
    num_reqs: int,
    max_tokens_per_req: int,
    block_size_kernel: int,
) -> int:
    """Choose the request-local worker count for the Ascend VectorCore path.

    Keep at least the upstream token-block parallelism so a balanced context
    range owned by one worker fits in BLOCK_SIZE. The VectorCore count is used
    as the target parallelism, not as a hard-coded device assumption.
    """
    upstream_num_blocks = triton.cdiv(
        max_tokens_per_req,
        block_size_kernel,
    )

    init_device_properties_triton()
    num_vectorcores = get_vectorcore_num()

    workers_for_vectorcores = max(
        1,
        triton.cdiv(num_vectorcores, num_reqs),
    )
    return max(
        upstream_num_blocks,
        workers_for_vectorcores,
    )


@triton.jit
def _cp_local_slot(
    positions,
    block_numbers,
    block_size,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
):
    """Return rank-local KV slots for the DCP DFlash contract.

    This is the same slot mapping rule used by current upstream vLLM. Keeping
    the helper local lets this module remain importable with older vLLM
    versions where cp_local_slot is not available.
    """
    block_offsets = positions % (block_size * CP_SIZE)
    if CP_SIZE == 1:
        return block_numbers * block_size + block_offsets

    is_local = (block_offsets // CP_INTERLEAVE % CP_SIZE) == cp_rank
    rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
    remainder = block_offsets % CP_INTERLEAVE
    local_offsets = rounds * CP_INTERLEAVE + remainder

    return tl.where(
        is_local,
        block_numbers * block_size + local_offsets,
        PAD_ID,
    )


# ============================================================================
# Non-DCP / legacy DFlash contract
# ============================================================================


@triton.jit
def _prepare_dflash_inputs_kernel_ascend(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
):
    """Balanced Ascend implementation for the non-DCP DFlash contract.

    The logical domains are partitioned independently:
      * context: request-local quotient/remainder partition
      * query: request-local quotient/remainder partition
      * sample: request-local quotient/remainder partition
      * graph padding: quotient/remainder partition over the whole launch grid

    This keeps the original contract while removing the one-program-per-request
    serialization of the previous Ascend implementation.
    """
    req_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)

    num_reqs = tl.num_programs(0)
    workers_per_req = tl.num_programs(1)

    global_worker_idx = req_idx * workers_per_req + worker_idx
    total_workers = num_reqs * workers_per_req

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    # ------------------------------------------------------------------
    # Context positions / slots.
    # ------------------------------------------------------------------
    ctx_base = num_ctx // workers_per_req
    ctx_extra = num_ctx % workers_per_req
    ctx_begin = worker_idx * ctx_base + tl.minimum(worker_idx, ctx_extra)
    ctx_count = ctx_base + tl.where(worker_idx < ctx_extra, 1, 0)

    ctx_lane = tl.arange(0, BLOCK_SIZE)
    ctx_mask = ctx_lane < ctx_count
    ctx_pos_idx = ctx_start + ctx_begin + ctx_lane

    ctx_pos = tl.load(
        target_positions_ptr + ctx_pos_idx,
        mask=ctx_mask,
        other=0,
    )
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(
        ctx_block_num,
        block_table_stride - 1,
    )
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=ctx_mask,
        other=0,
    ).to(tl.int64)

    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)

    tl.store(
        out_context_positions_ptr + ctx_pos_idx,
        ctx_pos,
        mask=ctx_mask,
    )
    tl.store(
        out_context_slot_mapping_ptr + ctx_pos_idx,
        ctx_slot,
        mask=ctx_mask,
    )

    # ------------------------------------------------------------------
    # Query positions / input_ids / slots.
    # ------------------------------------------------------------------
    query_base_count = num_query_per_req // workers_per_req
    query_extra = num_query_per_req % workers_per_req
    query_begin = worker_idx * query_base_count + tl.minimum(worker_idx, query_extra)
    query_count = query_base_count + tl.where(worker_idx < query_extra, 1, 0)

    for local_base in range(
        0,
        query_count,
        QUERY_BLOCK_SIZE,
    ):
        lane = tl.arange(0, QUERY_BLOCK_SIZE)
        local = local_base + lane
        mask = local < query_count
        query_off = query_begin + local

        query_pos = last_valid_pos + 1 + query_off
        query_idx = query_base + query_off

        input_id = tl.where(
            query_off == 0,
            bonus_token,
            parallel_drafting_token_id,
        )

        q_block_num = query_pos // block_size
        q_block_num = tl.minimum(
            q_block_num,
            block_table_stride - 1,
        )
        q_block_id = tl.load(
            block_table_ptr + req_idx * block_table_stride + q_block_num,
            mask=mask,
            other=0,
        ).to(tl.int64)

        q_slot = q_block_id * block_size + (query_pos % block_size)

        tl.store(
            out_input_ids_ptr + query_idx,
            input_id,
            mask=mask,
        )
        tl.store(
            out_query_positions_ptr + query_idx,
            tl.minimum(
                query_pos,
                max_model_len - 1,
            ),
            mask=mask,
        )
        tl.store(
            out_query_slot_mapping_ptr + query_idx,
            q_slot,
            mask=mask,
        )

    # ------------------------------------------------------------------
    # Sample indices / positions / idx_mapping.
    # ------------------------------------------------------------------
    sample_base_count = num_speculative_steps // workers_per_req
    sample_extra = num_speculative_steps % workers_per_req
    sample_begin = worker_idx * sample_base_count + tl.minimum(worker_idx, sample_extra)
    sample_count = sample_base_count + tl.where(worker_idx < sample_extra, 1, 0)

    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1

    for local_base in range(
        0,
        sample_count,
        QUERY_BLOCK_SIZE,
    ):
        lane = tl.arange(0, QUERY_BLOCK_SIZE)
        local = local_base + lane
        mask = local < sample_count

        sample_local = sample_begin + local
        query_off = sample_local + sample_off

        sample_idx = req_idx * num_speculative_steps + sample_local
        query_idx = query_base + query_off
        query_pos = last_valid_pos + 1 + query_off
        sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos

        tl.store(
            out_sample_indices_ptr + sample_idx,
            query_idx,
            mask=mask,
        )
        tl.store(
            out_sample_pos_ptr + sample_idx,
            sample_pos,
            mask=mask,
        )
        tl.store(
            out_sample_idx_mapping_ptr + sample_idx,
            req_state_idx,
            mask=mask,
        )

    # One owner for per-request scalar outputs.
    if worker_idx == 0:
        tl.store(
            out_query_start_loc_ptr + req_idx,
            query_base,
        )
        tl.store(
            out_seq_lens_ptr + req_idx,
            last_valid_pos + 1 + num_query_per_req,
        )
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(
            out_seeds_ptr + req_state_idx,
            tl.load(seeds_ptr + req_state_idx),
        )

    # ------------------------------------------------------------------
    # Graph padding.
    # ------------------------------------------------------------------
    last_query_end = num_reqs * num_query_per_req

    # query_start_loc:
    # [num_reqs, max_num_reqs + 1)
    qsl_pad_count = max_num_reqs + 1 - num_reqs
    qsl_base = qsl_pad_count // total_workers
    qsl_extra = qsl_pad_count % total_workers
    qsl_begin = global_worker_idx * qsl_base + tl.minimum(
        global_worker_idx,
        qsl_extra,
    )
    qsl_count = qsl_base + tl.where(
        global_worker_idx < qsl_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        qsl_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < qsl_count
        offset = num_reqs + qsl_begin + local

        tl.store(
            out_query_start_loc_ptr + offset,
            last_query_end,
            mask=mask,
        )

    # seq_lens:
    # [num_reqs, max_num_reqs)
    seq_pad_count = max_num_reqs - num_reqs
    seq_base = seq_pad_count // total_workers
    seq_extra = seq_pad_count % total_workers
    seq_begin = global_worker_idx * seq_base + tl.minimum(
        global_worker_idx,
        seq_extra,
    )
    seq_count = seq_base + tl.where(
        global_worker_idx < seq_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        seq_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < seq_count
        offset = num_reqs + seq_begin + local

        tl.store(
            out_seq_lens_ptr + offset,
            0,
            mask=mask,
        )

    # sample buffers:
    # [num_reqs * steps, max_num_reqs * steps)
    sample_pad_start = num_reqs * num_speculative_steps
    sample_pad_count = (max_num_reqs - num_reqs) * num_speculative_steps
    sample_pad_base = sample_pad_count // total_workers
    sample_pad_extra = sample_pad_count % total_workers
    sample_pad_begin = global_worker_idx * sample_pad_base + tl.minimum(
        global_worker_idx,
        sample_pad_extra,
    )
    sample_pad_worker_count = sample_pad_base + tl.where(
        global_worker_idx < sample_pad_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        sample_pad_worker_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < sample_pad_worker_count
        offset = sample_pad_start + sample_pad_begin + local

        tl.store(
            out_sample_indices_ptr + offset,
            0,
            mask=mask,
        )
        tl.store(
            out_sample_pos_ptr + offset,
            0,
            mask=mask,
        )
        tl.store(
            out_sample_idx_mapping_ptr + offset,
            -1,
            mask=mask,
        )

    # query_slot_mapping:
    # [num_reqs * num_query_per_req,
    #  max_num_tokens)
    query_pad_start = num_reqs * num_query_per_req
    query_pad_count = max_num_tokens - query_pad_start
    query_pad_base = query_pad_count // total_workers
    query_pad_extra = query_pad_count % total_workers
    query_pad_begin = global_worker_idx * query_pad_base + tl.minimum(
        global_worker_idx,
        query_pad_extra,
    )
    query_pad_worker_count = query_pad_base + tl.where(
        global_worker_idx < query_pad_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        query_pad_worker_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < query_pad_worker_count
        offset = query_pad_start + query_pad_begin + local

        tl.store(
            out_query_slot_mapping_ptr + offset,
            PAD_SLOT_ID,
            mask=mask,
        )


def prepare_dflash_inputs_ascend(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs]
    input_temperature: torch.Tensor,
    # [max_num_reqs]
    input_seeds: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    """Ascend replacement for the non-DCP prepare_dflash_inputs ABI."""
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0

    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req

    block_size_kernel = min(
        256,
        triton.next_power_of_2(max(1, max_tokens_per_req)),
    )

    workers_per_req = _get_workers_per_req(
        num_reqs,
        max_tokens_per_req,
        block_size_kernel,
    )

    _prepare_dflash_inputs_kernel_ascend[(num_reqs, workers_per_req)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=block_size_kernel,
        QUERY_BLOCK_SIZE=_QUERY_BLOCK_SIZE,
    )


# ============================================================================
# Current DCP-aware DFlash contract
# ============================================================================


@triton.jit
def _prepare_dflash_inputs_kernel_ascend_dcp(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
):
    """Balanced Ascend implementation for the current DCP-aware contract.

    In addition to the balanced program mapping, this preserves the current
    upstream DFlash semantics:
      * rejected context suffix is initialized to position=0/PAD_SLOT_ID
      * physical block 0 is treated as the null block
      * context/query slots use DCP rank-local slot mapping
      * seq_lens is clamped to max_model_len
    """
    req_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)

    num_reqs = tl.num_programs(0)
    workers_per_req = tl.num_programs(1)

    global_worker_idx = req_idx * workers_per_req + worker_idx
    total_workers = num_reqs * workers_per_req

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    # ------------------------------------------------------------------
    # Context positions / slots.
    #
    # Partition the full [0, num_ctx) output span. Rejected suffix rows are
    # still written, but as position=0/PAD_SLOT_ID exactly like upstream.
    # ------------------------------------------------------------------
    ctx_base = num_ctx // workers_per_req
    ctx_extra = num_ctx % workers_per_req
    ctx_begin = worker_idx * ctx_base + tl.minimum(worker_idx, ctx_extra)
    ctx_count = ctx_base + tl.where(worker_idx < ctx_extra, 1, 0)

    ctx_lane = tl.arange(0, BLOCK_SIZE)
    ctx_mask = ctx_lane < ctx_count
    ctx_off = ctx_begin + ctx_lane
    valid_ctx_mask = ctx_mask & (ctx_off < num_valid_ctx)
    ctx_pos_idx = ctx_start + ctx_off

    ctx_pos = tl.load(
        target_positions_ptr + ctx_pos_idx,
        mask=valid_ctx_mask,
        other=0,
    )

    ctx_block_num = ctx_pos // (block_size * CP_SIZE)
    ctx_block_num = tl.minimum(
        ctx_block_num,
        block_table_stride - 1,
    )
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=valid_ctx_mask,
        other=0,
    ).to(tl.int64)

    ctx_resident = valid_ctx_mask & (ctx_block_id != 0)

    local_ctx_slot = _cp_local_slot(
        ctx_pos,
        ctx_block_id,
        block_size,
        cp_rank,
        CP_SIZE,
        CP_INTERLEAVE,
        PAD_SLOT_ID,
    )
    ctx_slot = tl.where(
        ctx_resident,
        local_ctx_slot,
        PAD_SLOT_ID,
    )

    tl.store(
        out_context_positions_ptr + ctx_pos_idx,
        ctx_pos,
        mask=ctx_mask,
    )
    tl.store(
        out_context_slot_mapping_ptr + ctx_pos_idx,
        ctx_slot,
        mask=ctx_mask,
    )

    # ------------------------------------------------------------------
    # Query positions / input_ids / slots.
    # ------------------------------------------------------------------
    query_base_count = num_query_per_req // workers_per_req
    query_extra = num_query_per_req % workers_per_req
    query_begin = worker_idx * query_base_count + tl.minimum(worker_idx, query_extra)
    query_count = query_base_count + tl.where(worker_idx < query_extra, 1, 0)

    for local_base in range(
        0,
        query_count,
        QUERY_BLOCK_SIZE,
    ):
        lane = tl.arange(0, QUERY_BLOCK_SIZE)
        local = local_base + lane
        mask = local < query_count
        query_off = query_begin + local

        query_pos = last_valid_pos + 1 + query_off
        query_idx = query_base + query_off

        input_id = tl.where(
            query_off == 0,
            bonus_token,
            parallel_drafting_token_id,
        )

        q_block_num = query_pos // (block_size * CP_SIZE)
        q_block_num = tl.minimum(
            q_block_num,
            block_table_stride - 1,
        )
        q_block_id = tl.load(
            block_table_ptr + req_idx * block_table_stride + q_block_num,
            mask=mask,
            other=0,
        ).to(tl.int64)

        q_resident = mask & (q_block_id != 0)

        local_q_slot = _cp_local_slot(
            query_pos,
            q_block_id,
            block_size,
            cp_rank,
            CP_SIZE,
            CP_INTERLEAVE,
            PAD_SLOT_ID,
        )
        q_slot = tl.where(
            q_resident,
            local_q_slot,
            PAD_SLOT_ID,
        )

        tl.store(
            out_input_ids_ptr + query_idx,
            input_id,
            mask=mask,
        )
        tl.store(
            out_query_positions_ptr + query_idx,
            tl.minimum(
                query_pos,
                max_model_len - 1,
            ),
            mask=mask,
        )
        tl.store(
            out_query_slot_mapping_ptr + query_idx,
            q_slot,
            mask=mask,
        )

    # ------------------------------------------------------------------
    # Sample indices / positions / idx_mapping.
    # ------------------------------------------------------------------
    sample_base_count = num_speculative_steps // workers_per_req
    sample_extra = num_speculative_steps % workers_per_req
    sample_begin = worker_idx * sample_base_count + tl.minimum(worker_idx, sample_extra)
    sample_count = sample_base_count + tl.where(worker_idx < sample_extra, 1, 0)

    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1

    for local_base in range(
        0,
        sample_count,
        QUERY_BLOCK_SIZE,
    ):
        lane = tl.arange(0, QUERY_BLOCK_SIZE)
        local = local_base + lane
        mask = local < sample_count

        sample_local = sample_begin + local
        query_off = sample_local + sample_off

        sample_idx = req_idx * num_speculative_steps + sample_local
        query_idx = query_base + query_off
        query_pos = last_valid_pos + 1 + query_off
        sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos

        tl.store(
            out_sample_indices_ptr + sample_idx,
            query_idx,
            mask=mask,
        )
        tl.store(
            out_sample_pos_ptr + sample_idx,
            sample_pos,
            mask=mask,
        )
        tl.store(
            out_sample_idx_mapping_ptr + sample_idx,
            req_state_idx,
            mask=mask,
        )

    if worker_idx == 0:
        tl.store(
            out_query_start_loc_ptr + req_idx,
            query_base,
        )
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(
                last_valid_pos + 1 + num_query_per_req,
                max_model_len,
            ),
        )
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(
            out_seeds_ptr + req_state_idx,
            tl.load(seeds_ptr + req_state_idx),
        )

    # ------------------------------------------------------------------
    # Graph padding.
    # ------------------------------------------------------------------
    last_query_end = num_reqs * num_query_per_req

    qsl_pad_count = max_num_reqs + 1 - num_reqs
    qsl_base = qsl_pad_count // total_workers
    qsl_extra = qsl_pad_count % total_workers
    qsl_begin = global_worker_idx * qsl_base + tl.minimum(
        global_worker_idx,
        qsl_extra,
    )
    qsl_count = qsl_base + tl.where(
        global_worker_idx < qsl_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        qsl_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < qsl_count
        offset = num_reqs + qsl_begin + local

        tl.store(
            out_query_start_loc_ptr + offset,
            last_query_end,
            mask=mask,
        )

    seq_pad_count = max_num_reqs - num_reqs
    seq_base = seq_pad_count // total_workers
    seq_extra = seq_pad_count % total_workers
    seq_begin = global_worker_idx * seq_base + tl.minimum(
        global_worker_idx,
        seq_extra,
    )
    seq_count = seq_base + tl.where(
        global_worker_idx < seq_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        seq_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < seq_count
        offset = num_reqs + seq_begin + local

        tl.store(
            out_seq_lens_ptr + offset,
            0,
            mask=mask,
        )

    sample_pad_start = num_reqs * num_speculative_steps
    sample_pad_count = (max_num_reqs - num_reqs) * num_speculative_steps
    sample_pad_base = sample_pad_count // total_workers
    sample_pad_extra = sample_pad_count % total_workers
    sample_pad_begin = global_worker_idx * sample_pad_base + tl.minimum(
        global_worker_idx,
        sample_pad_extra,
    )
    sample_pad_worker_count = sample_pad_base + tl.where(
        global_worker_idx < sample_pad_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        sample_pad_worker_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < sample_pad_worker_count
        offset = sample_pad_start + sample_pad_begin + local

        tl.store(
            out_sample_indices_ptr + offset,
            0,
            mask=mask,
        )
        tl.store(
            out_sample_pos_ptr + offset,
            0,
            mask=mask,
        )
        tl.store(
            out_sample_idx_mapping_ptr + offset,
            -1,
            mask=mask,
        )

    query_pad_start = num_reqs * num_query_per_req
    query_pad_count = max_num_tokens - query_pad_start
    query_pad_base = query_pad_count // total_workers
    query_pad_extra = query_pad_count % total_workers
    query_pad_begin = global_worker_idx * query_pad_base + tl.minimum(
        global_worker_idx,
        query_pad_extra,
    )
    query_pad_worker_count = query_pad_base + tl.where(
        global_worker_idx < query_pad_extra,
        1,
        0,
    )

    for local_base in range(
        0,
        query_pad_worker_count,
        BLOCK_SIZE,
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        local = local_base + lane
        mask = local < query_pad_worker_count
        offset = query_pad_start + query_pad_begin + local

        tl.store(
            out_query_slot_mapping_ptr + offset,
            PAD_SLOT_ID,
            mask=mask,
        )


def prepare_dflash_inputs_ascend_dcp(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs]
    input_temperature: torch.Tensor,
    # [max_num_reqs]
    input_seeds: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    """Ascend replacement for the current DCP-aware prepare_dflash_inputs."""
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0

    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req

    block_size_kernel = min(
        256,
        triton.next_power_of_2(max(1, max_tokens_per_req)),
    )

    workers_per_req = _get_workers_per_req(
        num_reqs,
        max_tokens_per_req,
        block_size_kernel,
    )

    _prepare_dflash_inputs_kernel_ascend_dcp[(num_reqs, workers_per_req)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        BLOCK_SIZE=block_size_kernel,
        QUERY_BLOCK_SIZE=_QUERY_BLOCK_SIZE,
    )


__all__ = [
    "_prepare_dflash_inputs_kernel_ascend",
    "prepare_dflash_inputs_ascend",
    "_prepare_dflash_inputs_kernel_ascend_dcp",
    "prepare_dflash_inputs_ascend_dcp",
]
