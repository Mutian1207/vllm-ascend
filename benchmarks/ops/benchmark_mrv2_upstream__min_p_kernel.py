# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm_ascend.worker.v2.sample.min_p import apply_min_p


def case_min_p(device):
    for num_tokens, vocab in ((16, 32000), (64, 32000)):
        logits = torch.randn((num_tokens, vocab), device=device)
        idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
        min_p = torch.full((num_tokens,), 0.05, device=device)
        yield f"num_tokens={num_tokens} vocab_size={vocab}", lambda: apply_min_p(logits, idx, min_p), lambda: float(logits.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_min_p(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_min_p_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
