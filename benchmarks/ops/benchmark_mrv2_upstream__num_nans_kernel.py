# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import time

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm.v1.worker.gpu.metrics.logits import get_num_nans


def bench(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        out = fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    init_device_properties_triton()
    for num_reqs, vocab_size in [(4, 32_000), (32, 151_936)]:
        logits = torch.randn((num_reqs, vocab_size), device=args.device)
        logits[0, 0] = float("nan")
        latency_us, out = bench(lambda: get_num_nans(logits), args.warmup, args.repeat)
        print(f"op=_num_nans_kernel num_reqs={num_reqs} vocab_size={vocab_size} "
              f"latency_us={latency_us:.2f} checksum={int(out.sum().item())}")


if __name__ == "__main__":
    main()
