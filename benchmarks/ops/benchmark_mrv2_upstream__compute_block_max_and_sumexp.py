# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import time

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_block_max_and_sumexp,
)


@triton.jit
def _bench_kernel(out_max, out_sumexp, logits, stride, vocab_size,
                  BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    x = tl.load(logits + row * stride + block, mask=mask, other=float("-inf")).to(tl.float32)
    m, s = _compute_block_max_and_sumexp(x)
    bid = tl.program_id(1)
    tl.store(out_max + row * tl.num_programs(1) + bid, m)
    tl.store(out_sumexp + row * tl.num_programs(1) + bid, s)


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
    for num_logits, vocab_size in [(64, 32_000), (64, 151_936)]:
        block_size = 8192
        num_blocks = triton.cdiv(vocab_size, block_size)
        logits = torch.randn((num_logits, vocab_size), device=args.device)
        out_max = torch.empty((num_logits, num_blocks), device=args.device)
        out_sumexp = torch.empty_like(out_max)
        fn = lambda: _bench_kernel[(num_logits, num_blocks)](
            out_max, out_sumexp, logits, logits.stride(0), vocab_size,
            BLOCK_SIZE=block_size)
        latency_us = bench(fn, args.warmup, args.repeat)
        print(f"op=_compute_block_max_and_sumexp num_logits={num_logits} "
              f"vocab_size={vocab_size} latency_us={latency_us:.2f} "
              f"checksum={float(out_max.sum().item()) + float(out_sumexp.sum().item()):.3f}")


if __name__ == "__main__":
    main()
