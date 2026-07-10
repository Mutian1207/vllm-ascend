# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias


def case_bias(device):
    for num_tokens, vocab, n in ((16, 32000, 4), (64, 32000, 8)):
        logits = torch.randn((num_tokens, vocab), device=device)
        idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
        pos = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        zeros = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        num_bias = torch.full((num_tokens,), n, device=device, dtype=torch.int32)
        ids = torch.randint(0, vocab, (num_tokens, n), device=device, dtype=torch.int32)
        bias = torch.randn((num_tokens, n), device=device)
        min_lens = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        yield f"num_tokens={num_tokens} vocab_size={vocab} num_bias={n}", lambda: apply_logit_bias(logits, idx, pos, zeros, ids, num_bias, ids, bias, min_lens, zeros, ids), lambda: float(logits.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_bias(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_bias_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
