import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import (
    prepare_dflash_inputs_ascend,
    prepare_dflash_inputs_ascend_dcp,
)

BUSINESS_CASES = [
    {
        "name": "dflash_b8_t1626",
        "req_lens": [192, 197, 201, 205, 209, 211, 215, 196],
        "position_starts": [0, 256, 512, 768, 1024, 1280, 1536, 1792],
        "idx_mapping": [7, 1, 12, 3, 20, 5, 31, 9],
        "max_num_reqs": 64,
        "max_num_tokens": 8192,
        "max_model_len": 8192,
        "block_size": 128,
        "num_query_per_req": 9,
        "num_speculative_steps": 8,
        "parallel_drafting_token_id": 151669,
    },
    {
        "name": "dflash_b64_t576",
        "req_lens": [9] * 64,
        "position_starts": [i * 64 for i in range(64)],
        "idx_mapping": list(range(64)),
        "max_num_reqs": 64,
        "max_num_tokens": 8192,
        "max_model_len": 8192,
        "block_size": 128,
        "num_query_per_req": 9,
        "num_speculative_steps": 8,
        "parallel_drafting_token_id": 151669,
    },
]


def _build_query_start_loc(req_lens):
    values = [0]
    for length in req_lens:
        values.append(values[-1] + length)
    return values


def _build_positions(req_lens, position_starts):
    values = []
    for length, start in zip(req_lens, position_starts):
        values.extend(range(start, start + length))
    return values


def _allocate_outputs(max_num_reqs, max_num_tokens, num_speculative_steps, device):
    input_buffers = SimpleNamespace(
        input_ids=torch.full(
            (max_num_tokens,),
            -12345,
            dtype=torch.int32,
            device=device,
        ),
        positions=torch.full(
            (max_num_tokens,),
            -12345,
            dtype=torch.int64,
            device=device,
        ),
        query_start_loc=torch.full(
            (max_num_reqs + 1,),
            -12345,
            dtype=torch.int32,
            device=device,
        ),
        seq_lens=torch.full(
            (max_num_reqs,),
            -12345,
            dtype=torch.int32,
            device=device,
        ),
    )

    sample_capacity = max_num_reqs * num_speculative_steps
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=torch.full(
            (max_num_tokens,),
            -12345,
            dtype=torch.int32,
            device=device,
        ),
        context_positions=torch.full(
            (max_num_tokens,),
            -12345,
            dtype=torch.int64,
            device=device,
        ),
        context_slot_mapping=torch.full(
            (max_num_tokens,),
            -12345,
            dtype=torch.int32,
            device=device,
        ),
        sample_indices=torch.full(
            (sample_capacity,),
            -12345,
            dtype=torch.int64,
            device=device,
        ),
        sample_pos=torch.full(
            (sample_capacity,),
            -12345,
            dtype=torch.int64,
            device=device,
        ),
        sample_idx_mapping=torch.full(
            (sample_capacity,),
            -12345,
            dtype=torch.int32,
            device=device,
        ),
        temperature=torch.full(
            (max_num_reqs,),
            float("nan"),
            dtype=torch.float32,
            device=device,
        ),
        seeds=torch.full(
            (max_num_reqs,),
            -12345,
            dtype=torch.int64,
            device=device,
        ),
    )


def _build_business_inputs(case, device):
    req_lens = case["req_lens"]
    num_reqs = len(req_lens)
    max_num_reqs = case["max_num_reqs"]

    query_start_loc = _build_query_start_loc(req_lens)
    positions = _build_positions(req_lens, case["position_starts"])

    idx_mapping = torch.zeros(
        (max_num_reqs,),
        dtype=torch.int32,
        device=device,
    )
    idx_mapping[:num_reqs] = torch.tensor(
        case["idx_mapping"],
        dtype=torch.int32,
        device=device,
    )

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.asarray(req_lens, dtype=np.int32),
        positions=torch.tensor(
            positions,
            dtype=torch.int64,
            device=device,
        ),
        query_start_loc=torch.tensor(
            query_start_loc,
            dtype=torch.int32,
            device=device,
        ),
        idx_mapping=idx_mapping,
    )

    num_sampled = torch.ones(
        (num_reqs,),
        dtype=torch.int32,
        device=device,
    )
    num_rejected = torch.zeros(
        (num_reqs,),
        dtype=torch.int32,
        device=device,
    )

    last_sampled = torch.arange(
        1000,
        1000 + max_num_reqs,
        dtype=torch.int64,
        device=device,
    ).view(max_num_reqs, 1)
    next_prefill_tokens = torch.arange(
        2000,
        2000 + max_num_reqs,
        dtype=torch.int32,
        device=device,
    ).view(1, max_num_reqs)

    input_temperature = torch.linspace(
        0.5,
        1.5,
        max_num_reqs,
        dtype=torch.float32,
        device=device,
    )
    input_seeds = torch.arange(
        10000,
        10000 + max_num_reqs,
        dtype=torch.int64,
        device=device,
    )

    block_table = torch.arange(
        1,
        max_num_reqs * 64 + 1,
        dtype=torch.int32,
        device=device,
    ).view(max_num_reqs, 64)

    outputs = _allocate_outputs(
        max_num_reqs,
        case["max_num_tokens"],
        case["num_speculative_steps"],
        device,
    )

    return SimpleNamespace(
        input_batch=input_batch,
        outputs=outputs,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        next_prefill_tokens=next_prefill_tokens,
        input_temperature=input_temperature,
        input_seeds=input_seeds,
        block_table=block_table,
    )


def _cp_local_slot_ref(
    position,
    block_number,
    block_size,
    cp_rank,
    cp_size,
    cp_interleave,
):
    block_offset = position % (block_size * cp_size)
    if cp_size == 1:
        return block_number * block_size + block_offset

    is_local = (block_offset // cp_interleave) % cp_size == cp_rank
    if not is_local:
        return PAD_SLOT_ID

    rounds = block_offset // (cp_interleave * cp_size)
    remainder = block_offset % cp_interleave
    local_offset = rounds * cp_interleave + remainder
    return block_number * block_size + local_offset


def _build_reference(
    data,
    case,
    *,
    dcp,
    cp_rank=0,
    cp_size=1,
    cp_interleave=1,
):
    num_reqs = data.input_batch.num_reqs
    max_num_reqs = case["max_num_reqs"]
    max_num_tokens = case["max_num_tokens"]
    num_query_per_req = case["num_query_per_req"]
    num_speculative_steps = case["num_speculative_steps"]
    block_size = case["block_size"]
    max_model_len = case["max_model_len"]

    positions = data.input_batch.positions.cpu().tolist()
    query_start_loc = data.input_batch.query_start_loc.cpu().tolist()
    idx_mapping = data.input_batch.idx_mapping.cpu().tolist()
    num_sampled = data.num_sampled.cpu().tolist()
    num_rejected = data.num_rejected.cpu().tolist()
    last_sampled = data.last_sampled.view(-1).cpu().tolist()
    next_prefill_tokens = data.next_prefill_tokens.view(-1).cpu().tolist()
    input_temperature = data.input_temperature.cpu().tolist()
    input_seeds = data.input_seeds.cpu().tolist()
    block_table = data.block_table.cpu().tolist()

    ref = SimpleNamespace(
        input_ids=[None] * (num_reqs * num_query_per_req),
        query_positions=[None] * (num_reqs * num_query_per_req),
        query_start_loc=[None] * (max_num_reqs + 1),
        seq_lens=[None] * max_num_reqs,
        query_slot_mapping=[None] * max_num_tokens,
        context_positions=[None] * len(positions),
        context_slot_mapping=[None] * len(positions),
        sample_indices=[None] * (max_num_reqs * num_speculative_steps),
        sample_pos=[None] * (max_num_reqs * num_speculative_steps),
        sample_idx_mapping=[None] * (max_num_reqs * num_speculative_steps),
        temperature={},
        seeds={},
    )

    for req_idx in range(num_reqs):
        state_idx = idx_mapping[req_idx]
        ctx_start = query_start_loc[req_idx]
        ctx_end = query_start_loc[req_idx + 1]
        valid_ctx_end = ctx_end - num_rejected[req_idx]
        last_valid_pos = positions[valid_ctx_end - 1]

        if num_sampled[req_idx] > 0:
            bonus_token = last_sampled[state_idx]
        else:
            bonus_token = next_prefill_tokens[state_idx]

        for ctx_idx in range(ctx_start, ctx_end):
            if dcp and ctx_idx >= valid_ctx_end:
                ctx_pos = 0
                ctx_slot = PAD_SLOT_ID
            else:
                ctx_pos = positions[ctx_idx]
                block_divisor = block_size * cp_size if dcp else block_size
                logical_block = min(
                    ctx_pos // block_divisor,
                    len(block_table[req_idx]) - 1,
                )
                physical_block = block_table[req_idx][logical_block]

                if dcp:
                    if physical_block == 0:
                        ctx_slot = PAD_SLOT_ID
                    else:
                        ctx_slot = _cp_local_slot_ref(
                            ctx_pos,
                            physical_block,
                            block_size,
                            cp_rank,
                            cp_size,
                            cp_interleave,
                        )
                else:
                    ctx_slot = physical_block * block_size + ctx_pos % block_size

            ref.context_positions[ctx_idx] = ctx_pos
            ref.context_slot_mapping[ctx_idx] = ctx_slot

        query_base = req_idx * num_query_per_req
        ref.query_start_loc[req_idx] = query_base

        seq_len = last_valid_pos + 1 + num_query_per_req
        if dcp:
            seq_len = min(seq_len, max_model_len)
        ref.seq_lens[req_idx] = seq_len

        ref.temperature[state_idx] = input_temperature[state_idx]
        ref.seeds[state_idx] = input_seeds[state_idx]

        for query_off in range(num_query_per_req):
            query_idx = query_base + query_off
            query_pos = last_valid_pos + 1 + query_off
            ref.input_ids[query_idx] = bonus_token if query_off == 0 else case["parallel_drafting_token_id"]
            ref.query_positions[query_idx] = min(
                query_pos,
                max_model_len - 1,
            )

            block_divisor = block_size * cp_size if dcp else block_size
            logical_block = min(
                query_pos // block_divisor,
                len(block_table[req_idx]) - 1,
            )
            physical_block = block_table[req_idx][logical_block]

            if dcp:
                if physical_block == 0:
                    query_slot = PAD_SLOT_ID
                else:
                    query_slot = _cp_local_slot_ref(
                        query_pos,
                        physical_block,
                        block_size,
                        cp_rank,
                        cp_size,
                        cp_interleave,
                    )
            else:
                query_slot = physical_block * block_size + query_pos % block_size

            ref.query_slot_mapping[query_idx] = query_slot

        sample_off = 1
        for sample_local in range(num_speculative_steps):
            query_off = sample_local + sample_off
            sample_idx = req_idx * num_speculative_steps + sample_local
            query_idx = query_base + query_off
            query_pos = last_valid_pos + 1 + query_off

            ref.sample_indices[sample_idx] = query_idx
            ref.sample_pos[sample_idx] = query_pos
            ref.sample_idx_mapping[sample_idx] = state_idx

    last_query_end = num_reqs * num_query_per_req
    for i in range(num_reqs, max_num_reqs + 1):
        ref.query_start_loc[i] = last_query_end
    for i in range(num_reqs, max_num_reqs):
        ref.seq_lens[i] = 0

    sample_pad_start = num_reqs * num_speculative_steps
    sample_pad_end = max_num_reqs * num_speculative_steps
    for i in range(sample_pad_start, sample_pad_end):
        ref.sample_indices[i] = 0
        ref.sample_pos[i] = 0
        ref.sample_idx_mapping[i] = -1

    query_pad_start = num_reqs * num_query_per_req
    for i in range(query_pad_start, max_num_tokens):
        ref.query_slot_mapping[i] = PAD_SLOT_ID

    return ref


def _assert_exact(actual, expected):
    expected_tensor = torch.tensor(
        expected,
        dtype=actual.dtype,
        device=actual.device,
    )
    torch.testing.assert_close(
        actual,
        expected_tensor,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def _validate_outputs(data, case, ref):
    outputs = data.outputs
    num_reqs = data.input_batch.num_reqs
    total_context = int(data.input_batch.query_start_loc[num_reqs].item())
    total_query = num_reqs * case["num_query_per_req"]

    _assert_exact(
        outputs.input_buffers.input_ids[:total_query],
        ref.input_ids,
    )
    _assert_exact(
        outputs.input_buffers.positions[:total_query],
        ref.query_positions,
    )
    _assert_exact(
        outputs.input_buffers.query_start_loc,
        ref.query_start_loc,
    )
    _assert_exact(
        outputs.input_buffers.seq_lens,
        ref.seq_lens,
    )
    _assert_exact(
        outputs.query_slot_mapping,
        ref.query_slot_mapping,
    )
    _assert_exact(
        outputs.context_positions[:total_context],
        ref.context_positions,
    )
    _assert_exact(
        outputs.context_slot_mapping[:total_context],
        ref.context_slot_mapping,
    )
    _assert_exact(
        outputs.sample_indices,
        ref.sample_indices,
    )
    _assert_exact(
        outputs.sample_pos,
        ref.sample_pos,
    )
    _assert_exact(
        outputs.sample_idx_mapping,
        ref.sample_idx_mapping,
    )

    for state_idx, expected in ref.temperature.items():
        torch.testing.assert_close(
            outputs.temperature[state_idx],
            torch.tensor(
                expected,
                dtype=torch.float32,
                device=outputs.temperature.device,
            ),
            rtol=0,
            atol=0,
        )
    for state_idx, expected in ref.seeds.items():
        assert outputs.seeds[state_idx].item() == expected


@pytest.mark.parametrize("case", BUSINESS_CASES, ids=lambda case: case["name"])
def test_prepare_dflash_inputs_ascend_business(case):
    device = "npu"
    data = _build_business_inputs(case, device)

    prepare_dflash_inputs_ascend(
        data.outputs.input_buffers,
        data.outputs.query_slot_mapping,
        data.outputs.context_positions,
        data.outputs.context_slot_mapping,
        data.outputs.sample_indices,
        data.outputs.sample_pos,
        data.outputs.sample_idx_mapping,
        data.outputs.temperature,
        data.outputs.seeds,
        data.input_batch,
        data.num_sampled,
        data.num_rejected,
        data.last_sampled,
        data.next_prefill_tokens,
        data.input_temperature,
        data.input_seeds,
        data.block_table,
        case["block_size"],
        case["parallel_drafting_token_id"],
        case["num_query_per_req"],
        case["num_speculative_steps"],
        case["max_num_reqs"],
        case["max_num_tokens"],
        case["max_model_len"],
        False,
    )

    ref = _build_reference(data, case, dcp=False)
    _validate_outputs(data, case, ref)

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("case", BUSINESS_CASES, ids=lambda case: case["name"])
def test_prepare_dflash_inputs_ascend_dcp_cp1_business(case):
    device = "npu"
    data = _build_business_inputs(case, device)

    prepare_dflash_inputs_ascend_dcp(
        data.outputs.input_buffers,
        data.outputs.query_slot_mapping,
        data.outputs.context_positions,
        data.outputs.context_slot_mapping,
        data.outputs.sample_indices,
        data.outputs.sample_pos,
        data.outputs.sample_idx_mapping,
        data.outputs.temperature,
        data.outputs.seeds,
        data.input_batch,
        data.num_sampled,
        data.num_rejected,
        data.last_sampled,
        data.next_prefill_tokens,
        data.input_temperature,
        data.input_seeds,
        data.block_table,
        case["block_size"],
        0,
        1,
        1,
        case["parallel_drafting_token_id"],
        case["num_query_per_req"],
        case["num_speculative_steps"],
        case["max_num_reqs"],
        case["max_num_tokens"],
        case["max_model_len"],
        False,
    )

    ref = _build_reference(
        data,
        case,
        dcp=True,
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
    )
    _validate_outputs(data, case, ref)

    gc.collect()
    torch.npu.empty_cache()


def test_prepare_dflash_inputs_ascend_dcp_rejected_context_and_local_slot():
    device = "npu"
    case = {
        "name": "dcp_rejected_context",
        "req_lens": [4],
        "position_starts": [10],
        "idx_mapping": [2],
        "max_num_reqs": 4,
        "max_num_tokens": 16,
        "max_model_len": 64,
        "block_size": 4,
        "num_query_per_req": 3,
        "num_speculative_steps": 3,
        "parallel_drafting_token_id": 123,
    }

    data = _build_business_inputs(case, device)
    data.num_rejected.fill_(2)

    data.block_table.zero_()
    data.block_table[0, :4] = torch.tensor(
        [0, 7, 8, 9],
        dtype=torch.int32,
        device=device,
    )

    prepare_dflash_inputs_ascend_dcp(
        data.outputs.input_buffers,
        data.outputs.query_slot_mapping,
        data.outputs.context_positions,
        data.outputs.context_slot_mapping,
        data.outputs.sample_indices,
        data.outputs.sample_pos,
        data.outputs.sample_idx_mapping,
        data.outputs.temperature,
        data.outputs.seeds,
        data.input_batch,
        data.num_sampled,
        data.num_rejected,
        data.last_sampled,
        data.next_prefill_tokens,
        data.input_temperature,
        data.input_seeds,
        data.block_table,
        case["block_size"],
        1,
        2,
        2,
        case["parallel_drafting_token_id"],
        case["num_query_per_req"],
        case["num_speculative_steps"],
        case["max_num_reqs"],
        case["max_num_tokens"],
        case["max_model_len"],
        True,
    )

    assert data.outputs.context_positions[:4].cpu().tolist() == [10, 11, 0, 0]
    assert data.outputs.context_slot_mapping[:4].cpu().tolist() == [
        28,
        29,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert data.outputs.query_slot_mapping[:3].cpu().tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        30,
    ]

    gc.collect()
    torch.npu.empty_cache()
