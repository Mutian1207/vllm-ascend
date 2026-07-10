# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.input_batch import expand_idx_mapping


def case_expand_idx_mapping(device):
    for num_reqs, steps in ((16, 3), (128, 5)):
        total = num_reqs * steps
        idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        cu = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * steps
        holder = {}
        def fn():
            holder['out'] = expand_idx_mapping(idx, total, cu, steps)
            return holder['out']
        yield f"num_reqs={num_reqs} max_expand_len={steps}", fn, lambda: int(holder['out'][0].sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_expand_idx_mapping(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_expand_idx_mapping_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
