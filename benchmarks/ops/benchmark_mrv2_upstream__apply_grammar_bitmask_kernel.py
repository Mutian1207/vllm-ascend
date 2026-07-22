# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel


def case_grammar_bitmask(device):
    for rows, vocab in ((16, 32000), (64, 32000)):
        logits = torch.randn((rows, vocab), device=device)
        logits_indices = torch.arange(rows, device=device, dtype=torch.int64)
        bitmask = torch.full((rows, triton.cdiv(vocab, 32)), -1, device=device, dtype=torch.int32)
        grid = (rows, triton.cdiv(vocab, 1024))
        yield f"rows={rows} vocab_size={vocab}", lambda: _apply_grammar_bitmask_kernel[grid](logits, logits.stride(0), logits_indices, bitmask, bitmask.stride(0), vocab, BLOCK_SIZE=1024), lambda: float(logits.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_grammar_bitmask(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_apply_grammar_bitmask_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
