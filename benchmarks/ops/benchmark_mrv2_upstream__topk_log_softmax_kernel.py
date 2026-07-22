# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs


def case_topk_log_softmax(device):
    for batch, vocab, topk in ((16, 32000, 5), (64, 32000, 5)):
        logits = torch.randn((batch, vocab), device=device)
        token_ids = torch.randint(0, vocab, (batch, topk), device=device, dtype=torch.int64)
        holder = {}
        def fn():
            holder['out'] = compute_token_logprobs(logits, token_ids)
            return holder['out']
        yield f"batch_size={batch} vocab_size={vocab} topk={topk}", fn, lambda: float(holder['out'].sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_topk_log_softmax(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_topk_log_softmax_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
