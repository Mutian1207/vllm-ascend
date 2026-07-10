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
from vllm_ascend.worker.v2.sample.penalties import bincount


def case_bincount(device):
    for num_tokens, vocab, max_prefill in ((16, 32000, 64), (64, 32000, 128)):
        idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
        all_ids = torch.randint(0, vocab, (num_tokens, max_prefill), device=device, dtype=torch.int32)
        prompt_len = torch.full((num_tokens,), max_prefill // 2, device=device, dtype=torch.int32)
        prefill_len = torch.full((num_tokens,), max_prefill, device=device, dtype=torch.int32)
        prompt_mask = torch.zeros((num_tokens, triton.cdiv(vocab, 32)), device=device, dtype=torch.int32)
        counts = torch.zeros((num_tokens, vocab), device=device, dtype=torch.int32)
        yield f"num_tokens={num_tokens} vocab_size={vocab} max_prefill_len={max_prefill}", lambda: bincount(idx, all_ids, prompt_len, prefill_len, prompt_mask, counts, max_prefill), lambda: int(counts.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_bincount(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_bincount_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
