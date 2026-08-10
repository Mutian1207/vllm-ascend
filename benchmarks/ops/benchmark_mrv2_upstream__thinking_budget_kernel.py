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
from vllm.v1.worker.gpu.sample.thinking_budget import _thinking_budget_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    max_num_reqs = 16
    max_len = 2048
    vocab_size = 1024
    num_tokens = max_num_reqs
    start_len = 1
    natural_end_len = 1
    end_len = 1
    for total_len in [100, 500]:
        logits = torch.randn((num_tokens, vocab_size), device=args.device)
        expanded_idx_mapping = torch.arange(
            num_tokens, device=args.device, dtype=torch.int32
        )
        # budget >= 0 so the budget-application path runs.
        thinking_token_budget = torch.full(
            (max_num_reqs,), 32, device=args.device, dtype=torch.int32
        )
        all_token_ids = torch.randint(
            0, 32000, (max_num_reqs, max_len), device=args.device, dtype=torch.int32
        )
        total_len_t = torch.full(
            (max_num_reqs,), total_len, device=args.device, dtype=torch.int32
        )
        input_ids = torch.randint(
            0, vocab_size, (num_tokens,), device=args.device, dtype=torch.int32
        )
        expanded_local_pos = torch.zeros(num_tokens, device=args.device, dtype=torch.int32)
        # Seed cached markers so the force-token store path is exercised.
        cached_last_start = torch.full(
            (max_num_reqs,), 49, device=args.device, dtype=torch.int32
        )
        cached_last_end = torch.full(
            (max_num_reqs,), -1, device=args.device, dtype=torch.int32
        )
        reasoning_start_token_ids = torch.tensor(
            [1000], device=args.device, dtype=torch.int32
        )
        natural_reasoning_end_token_ids = torch.tensor(
            [2000], device=args.device, dtype=torch.int32
        )
        reasoning_end_token_ids = torch.tensor(
            [500], device=args.device, dtype=torch.int32
        )
        fn = lambda: _thinking_budget_kernel[(num_tokens,)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            thinking_token_budget,
            all_token_ids,
            all_token_ids.stride(0),
            total_len_t,
            input_ids,
            expanded_local_pos,
            cached_last_start,
            cached_last_end,
            reasoning_start_token_ids,
            natural_reasoning_end_token_ids,
            reasoning_end_token_ids,
            START_LEN=start_len,
            NATURAL_END_LEN=natural_end_len,
            END_LEN=end_len,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_thinking_budget_kernel num_tokens={num_tokens} "
            f"total_len={total_len} latency_us={latency_us:.2f} "
            f"checksum={float(logits.sum().item()):.3f}"
        )


if __name__ == "__main__":
    main()
