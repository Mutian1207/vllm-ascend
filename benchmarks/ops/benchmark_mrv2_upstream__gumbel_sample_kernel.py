# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample


def case_gumbel_sample(device):
    for num_tokens, vocab in ((16, 32000), (64, 32000)):
        logits = torch.randn((num_tokens, vocab), device=device)
        idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
        temp = torch.ones(num_tokens, device=device)
        seed = torch.arange(num_tokens, device=device, dtype=torch.int64)
        pos = torch.arange(num_tokens, device=device, dtype=torch.int32)
        holder = {}
        def fn():
            holder['out'] = gumbel_sample(logits, idx, temp, seed, pos, apply_temperature=False, use_fp64=False)
            return holder['out']
        yield f"num_tokens={num_tokens} vocab_size={vocab}", fn, lambda: int(holder['out'].sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_gumbel_sample(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_gumbel_sample_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
