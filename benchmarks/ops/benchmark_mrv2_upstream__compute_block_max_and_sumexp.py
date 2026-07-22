# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_block_max_and_sumexp,
)


@triton.jit
def _bench_kernel(out_max, out_sumexp, logits, stride, vocab_size,
                  BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    block = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    x = tl.load(logits + row * stride + block,
                mask=mask,
                other=float("-inf")).to(tl.float32)
    m, s = _compute_block_max_and_sumexp(x)
    tl.store(out_max + row * tl.num_programs(1) + block_id, m)
    tl.store(out_sumexp + row * tl.num_programs(1) + block_id, s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for num_logits, vocab_size in [(64, 32_000), (64, 151_936)]:
        block_size = 8192
        num_blocks = triton.cdiv(vocab_size, block_size)
        logits = torch.randn((num_logits, vocab_size), device=args.device)
        out_max = torch.empty((num_logits, num_blocks), device=args.device)
        out_sumexp = torch.empty_like(out_max)
        fn = lambda: _bench_kernel[(num_logits, num_blocks)](
            out_max, out_sumexp, logits, logits.stride(0), vocab_size,
            BLOCK_SIZE=block_size)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        checksum = float(out_max.sum().item()) + float(out_sumexp.sum().item())
        print(f"op=_compute_block_max_and_sumexp num_logits={num_logits} "
              f"vocab_size={vocab_size} latency_us={latency_us:.2f} "
              f"checksum={checksum:.3f}")


if __name__ == "__main__":
    main()
