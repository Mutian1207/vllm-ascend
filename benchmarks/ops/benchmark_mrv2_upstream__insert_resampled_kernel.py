# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import time

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _insert_resampled_kernel


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
    for num_reqs, steps, num_blocks in [(16, 1, 32), (64, 3, 32)]:
        sampled = torch.empty((num_reqs, steps + 1), device=args.device, dtype=torch.int64)
        num_sampled = torch.zeros(num_reqs, device=args.device, dtype=torch.int32)
        resampled_idx = torch.randint(0, 32_000, (num_reqs, num_blocks),
                                      device=args.device, dtype=torch.int64)
        resampled_max = torch.randn((num_reqs, num_blocks), device=args.device)
        cu_num_logits = torch.arange(num_reqs + 1, device=args.device,
                                     dtype=torch.int32) * (steps + 1)
        expanded_idx = torch.arange(num_reqs, device=args.device,
                                    dtype=torch.int32).repeat_interleave(steps + 1)
        temp = torch.ones(num_reqs, device=args.device)
        padded = triton.next_power_of_2(num_blocks)
        fn = lambda: _insert_resampled_kernel[(num_reqs,)](
            sampled, sampled.stride(0), num_sampled, resampled_idx,
            resampled_idx.stride(0), resampled_max, resampled_max.stride(0),
            num_blocks, cu_num_logits, expanded_idx, temp,
            PADDED_RESAMPLE_NUM_BLOCKS=padded)
        latency_us = bench(fn, args.warmup, args.repeat)
        print(f"op=_insert_resampled_kernel num_reqs={num_reqs} steps={steps} "
              f"num_blocks={num_blocks} latency_us={latency_us:.2f} "
              f"checksum={int(sampled.sum().item() + num_sampled.sum().item())}")


if __name__ == "__main__":
    main()
