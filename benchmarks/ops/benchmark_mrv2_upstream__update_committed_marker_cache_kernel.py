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
from vllm.v1.worker.gpu.sample.thinking_budget import (
    _update_committed_marker_cache_kernel,
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
    max_len = 2048
    start_len = 1
    natural_end_len = 1
    reasoning_start_token_ids = torch.tensor(
        [1000], device=args.device, dtype=torch.int32
    )
    natural_reasoning_end_token_ids = torch.tensor(
        [2000], device=args.device, dtype=torch.int32
    )
    for total_len in [50, 200, 1000]:
        req_ids = torch.arange(max_num_reqs, device=args.device, dtype=torch.int32)
        # budget >= 0 so the cold-scan path runs.
        thinking_token_budget = torch.full(
            (max_num_reqs,), 32, device=args.device, dtype=torch.int32
        )
        all_token_ids = torch.randint(
            0, 32000, (max_num_reqs, max_len), device=args.device, dtype=torch.int32
        )
        # Plant a start marker so the backward cold scan does real work.
        all_token_ids[:, total_len - 1] = 1000
        total_len_t = torch.full(
            (max_num_reqs,), total_len, device=args.device, dtype=torch.int32
        )
        cached_last_start = torch.full(
            (max_num_reqs,), -1, device=args.device, dtype=torch.int32
        )
        cached_last_end = torch.full(
            (max_num_reqs,), -1, device=args.device, dtype=torch.int32
        )
        cached_scan_pos = torch.full(
            (max_num_reqs,), 0, device=args.device, dtype=torch.int32
        )
        fn = lambda: _update_committed_marker_cache_kernel[(req_ids.shape[0],)](
            req_ids,
            thinking_token_budget,
            all_token_ids,
            all_token_ids.stride(0),
            total_len_t,
            cached_last_start,
            cached_last_end,
            cached_scan_pos,
            reasoning_start_token_ids,
            natural_reasoning_end_token_ids,
            START_LEN=start_len,
            NATURAL_END_LEN=natural_end_len,
            MAX_LEN=max(start_len, natural_end_len),
            BLOCK=1024,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_update_committed_marker_cache_kernel num_reqs={max_num_reqs} "
            f"total_len={total_len} latency_us={latency_us:.2f} "
            f"checksum={int(cached_last_start.sum().item())}"
        )


if __name__ == "__main__":
    main()
