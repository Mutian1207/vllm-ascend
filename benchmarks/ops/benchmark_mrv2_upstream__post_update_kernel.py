# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.input_batch import post_update


def case_post_update(device):
    for num_reqs, steps in ((16, 2), (128, 4)):
        max_len = 4096
        vocab_size = 32000
        idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        num_comp = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        last = torch.zeros(num_reqs, device=device, dtype=torch.int64)
        output_bin_counts = torch.zeros((num_reqs, vocab_size),
                                        device=device,
                                        dtype=torch.int32)
        sampled = torch.randint(0,
                                vocab_size,
                                (num_reqs, steps + 1),
                                device=device,
                                dtype=torch.int64)
        num_sampled = torch.full((num_reqs,),
                                 steps + 1,
                                 device=device,
                                 dtype=torch.int32)
        num_rejected = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        qsl = torch.arange(num_reqs + 1, device=device,
                           dtype=torch.int32) * (steps + 1)
        all_ids = torch.zeros((num_reqs, max_len),
                              device=device,
                              dtype=torch.int32)
        total_len = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        yield (
            f"num_reqs={num_reqs} steps={steps}",
            lambda: post_update(idx, num_comp, last, output_bin_counts, sampled,
                                num_sampled, num_rejected, qsl, all_ids,
                                total_len),
            lambda: int(total_len.sum().item()),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_post_update(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_post_update_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
