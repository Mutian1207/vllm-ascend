# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected


def case_get_num_sampled_and_rejected(device):
    for num_reqs, steps in ((16, 2), (128, 4)):
        num_sampled = torch.full((num_reqs,), steps, device=device, dtype=torch.int32)
        seq = torch.full((num_reqs,), 16, device=device, dtype=torch.int32)
        cu = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * (steps + 1)
        idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        prefill = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        holder = {}
        def fn():
            holder['out'] = get_num_sampled_and_rejected(num_sampled, seq, cu, idx, prefill)
            return holder['out']
        yield f"num_reqs={num_reqs} steps={steps}", fn, lambda: int(holder['out'][1].sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_get_num_sampled_and_rejected(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_get_num_sampled_and_rejected_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
