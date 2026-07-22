# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel


def case_ranks(device):
    for batch, vocab in ((16, 32000), (64, 32000)):
        logits = torch.randn((batch, vocab), device=device)
        token_ids = torch.randint(0, vocab, (batch,), device=device, dtype=torch.int64)
        out = torch.empty(batch, device=device, dtype=torch.int64)
        yield f"batch_size={batch} vocab_size={vocab}", lambda: _ranks_kernel[(batch,)](out, logits, logits.stride(0), token_ids, vocab, BLOCK_SIZE=8192), lambda: int(out.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_ranks(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_ranks_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
