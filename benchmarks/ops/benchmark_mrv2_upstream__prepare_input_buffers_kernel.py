# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    _prepare_input_buffers_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    max_num_reqs = 16
    num_speculative_steps = 3
    max_query_len = 256
    for num_reqs in [1, 16]:
        query_start_loc = torch.arange(
            0, (num_reqs + 1) * (num_speculative_steps + 1),
            num_speculative_steps + 1, device=args.device, dtype=torch.int32
        )
        last_token_indices = (query_start_loc[1:] - 1).clone()
        draft_input_ids = torch.zeros(max_query_len, device=args.device, dtype=torch.int32)
        positions = torch.zeros(max_query_len, device=args.device, dtype=torch.int32)
        seq_lens = torch.full((max_num_reqs,), 4, device=args.device, dtype=torch.int32)
        target_input_ids = torch.randint(
            0, 32000, (max_query_len,), device=args.device, dtype=torch.int32
        )
        target_positions = torch.arange(
            max_query_len, device=args.device, dtype=torch.int32
        )
        cached_draft_input_ids = torch.randint(
            0, 32000, (max_num_reqs, num_speculative_steps - 1),
            device=args.device, dtype=torch.int32
        )
        draft_input_id_overrides = torch.full(
            (max_num_reqs, num_speculative_steps - 1),
            -1, device=args.device, dtype=torch.int64
        )
        idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
        last_sampled = torch.randint(
            0, 32000, (max_num_reqs,), device=args.device, dtype=torch.int64
        )
        next_prefill_tokens = torch.randint(
            0, 32000, (num_speculative_steps - 1, max_num_reqs),
            device=args.device, dtype=torch.int32
        )
        num_sampled = torch.ones(num_reqs, device=args.device, dtype=torch.int32)
        num_rejected = torch.ones(num_reqs, device=args.device, dtype=torch.int32)
        fn = lambda: _prepare_input_buffers_kernel[(num_reqs,)](
            last_token_indices,
            draft_input_ids,
            positions,
            seq_lens,
            target_input_ids,
            target_positions,
            cached_draft_input_ids,
            cached_draft_input_ids.stride(0),
            draft_input_id_overrides,
            draft_input_id_overrides.stride(0),
            idx_mapping,
            last_sampled,
            next_prefill_tokens,
            next_prefill_tokens.stride(0),
            num_sampled,
            num_rejected,
            seq_lens,  # target_seq_lens_ptr (target seq lens, [max_num_reqs])
            query_start_loc,
            max_num_reqs,
            num_speculative_steps,
            BLOCK_SIZE=1024,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_prepare_input_buffers_kernel num_reqs={num_reqs} "
            f"num_speculative_steps={num_speculative_steps} "
            f"latency_us={latency_us:.2f} "
            f"checksum={int(last_token_indices.sum().item())}"
        )


if __name__ == "__main__":
    main()
