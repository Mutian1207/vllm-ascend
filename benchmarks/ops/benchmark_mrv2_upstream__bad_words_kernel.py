# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words


def case_bad_words(device):
    for num_tokens, vocab, max_bw in ((16, 32000, 2), (64, 32000, 4)):
        logits = torch.randn((num_tokens, vocab), device=device)
        idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
        bad_tokens = torch.randint(0, vocab, (num_tokens, 8), device=device, dtype=torch.int32)
        offsets = torch.tensor([[0, 2, 4, 6, 8]] * num_tokens, device=device, dtype=torch.int32)
        num_bw = torch.full((num_tokens,), max_bw, device=device, dtype=torch.int32)
        all_ids = torch.randint(0, vocab, (num_tokens, 64), device=device, dtype=torch.int32)
        prompt_len = torch.full((num_tokens,), 16, device=device, dtype=torch.int32)
        total_len = torch.full((num_tokens,), 32, device=device, dtype=torch.int32)
        input_ids = torch.randint(0, vocab, (num_tokens,), device=device, dtype=torch.int32)
        local_pos = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        yield f"num_tokens={num_tokens} vocab_size={vocab} max_bad_words={max_bw}", lambda: apply_bad_words(logits, idx, bad_tokens, offsets, num_bw, all_ids, prompt_len, total_len, input_ids, local_pos, max_bw), lambda: float(logits.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_bad_words(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_bad_words_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
