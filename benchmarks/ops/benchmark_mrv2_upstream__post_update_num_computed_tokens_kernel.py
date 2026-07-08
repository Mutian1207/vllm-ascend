# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import time

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm.v1.worker.gpu.input_batch import post_update_num_computed_tokens


def bench(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    init_device_properties_triton()
    for num_reqs in [16, 128, 512]:
        idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
        computed = torch.zeros(num_reqs, device=args.device, dtype=torch.int32)
        query_lens = torch.randint(1, 16, (num_reqs,), device=args.device, dtype=torch.int32)
        query_start_loc = torch.cat((torch.zeros((1,), device=args.device, dtype=torch.int32),
                                     torch.cumsum(query_lens, dim=0)))
        fn = lambda: post_update_num_computed_tokens(idx_mapping, computed, query_start_loc)
        latency_us = bench(fn, args.warmup, args.repeat)
        print(f"op=_post_update_num_computed_tokens_kernel num_reqs={num_reqs} "
              f"latency_us={latency_us:.2f} checksum={int(computed.sum().item())}")


if __name__ == "__main__":
    main()
