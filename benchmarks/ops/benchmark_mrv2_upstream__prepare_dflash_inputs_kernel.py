# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel,
)


def prepare_dflash_inputs_npu(
    input_buffers,
    query_slot_mapping,
    context_positions,
    context_slot_mapping,
    sample_indices,
    sample_pos,
    sample_idx_mapping,
    input_batch,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill_tokens,
    block_table,
    block_size,
    parallel_drafting_token_id,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
) -> None:
    num_reqs = input_batch.num_reqs
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    block_elems = min(256, triton.next_power_of_2(max(16, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, block_elems)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
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
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=block_elems,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for num_reqs, context_len in [(16, 8), (128, 1), (128, 16)]:
        num_query_per_req = 3
        num_speculative_steps = 2
        max_num_tokens = num_reqs * num_query_per_req
        total_context_tokens = num_reqs * context_len
        buffers = InputBuffers(num_reqs, max_num_tokens, torch.device(args.device))
        query_slot_mapping = torch.empty(max_num_tokens, device=args.device, dtype=torch.int64)
        context_positions = torch.empty(total_context_tokens, device=args.device,
                                        dtype=torch.int64)
        context_slot_mapping = torch.empty(total_context_tokens, device=args.device,
                                           dtype=torch.int64)
        sample_shape = (num_reqs * num_speculative_steps,)
        sample_indices = torch.empty(sample_shape, device=args.device, dtype=torch.int32)
        sample_pos = torch.empty(sample_shape, device=args.device, dtype=torch.int64)
        sample_idx_mapping = torch.empty(sample_shape, device=args.device, dtype=torch.int32)
        target_positions = torch.arange(total_context_tokens, device=args.device,
                                        dtype=torch.int64) + 8
        query_start_loc = torch.arange(num_reqs + 1, device=args.device,
                                       dtype=torch.int32) * context_len
        input_batch = SimpleNamespace(
            num_reqs=num_reqs,
            num_scheduled_tokens=np.full(num_reqs, context_len, dtype=np.int32),
            positions=target_positions,
            query_start_loc=query_start_loc,
            idx_mapping=torch.arange(num_reqs, device=args.device, dtype=torch.int32),
        )
        num_sampled = torch.ones(num_reqs, device=args.device, dtype=torch.int32)
        num_rejected = torch.zeros(num_reqs, device=args.device, dtype=torch.int32)
        last_sampled = torch.randint(0, 32_000, (num_reqs,), device=args.device)
        next_prefill = torch.randint(0, 32_000, (num_reqs,), device=args.device)
        block_table = torch.arange(num_reqs * 512, device=args.device,
                                   dtype=torch.int32).reshape(num_reqs, 512)
        fn = lambda: prepare_dflash_inputs_npu(
            buffers, query_slot_mapping, context_positions, context_slot_mapping,
            sample_indices, sample_pos, sample_idx_mapping, input_batch, num_sampled,
            num_rejected, last_sampled, next_prefill, block_table, 16, 99_999,
            num_query_per_req, num_speculative_steps, num_reqs, max_num_tokens)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_prepare_dflash_inputs_kernel num_reqs={num_reqs} "
              f"context_len={context_len} latency_us={latency_us:.2f} "
              f"checksum={int(buffers.input_ids.sum().item())}")


if __name__ == "__main__":
    main()
