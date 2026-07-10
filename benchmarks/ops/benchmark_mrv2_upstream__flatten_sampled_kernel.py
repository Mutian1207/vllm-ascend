# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import _flatten_sampled_kernel


def case_flatten_sampled(device):
    for num_reqs, steps in ((16, 2), (128, 4)):
        sampled = torch.randint(0, 32000, (num_reqs, steps + 1), device=device, dtype=torch.int64)
        num_sampled = torch.full((num_reqs,), steps + 1, device=device, dtype=torch.int32)
        cu = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * (steps + 1)
        flat = torch.empty(num_reqs * (steps + 1), device=device, dtype=torch.int64)
        yield f"num_reqs={num_reqs} steps={steps}", lambda: _flatten_sampled_kernel[(num_reqs,)](flat, sampled, sampled.stride(0), num_sampled, cu), lambda: int(flat.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_flatten_sampled(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_flatten_sampled_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
