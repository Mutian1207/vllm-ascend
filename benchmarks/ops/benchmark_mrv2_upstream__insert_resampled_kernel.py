# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _insert_resampled_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

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
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_insert_resampled_kernel num_reqs={num_reqs} steps={steps} "
              f"num_blocks={num_blocks} latency_us={latency_us:.2f} "
              f"checksum={int(sampled.sum().item() + num_sampled.sum().item())}")


if __name__ == "__main__":
    main()

