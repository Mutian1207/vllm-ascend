# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import time

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import prepare_decode_inputs


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
    for num_reqs in [16, 128, 512]:
        buffers = InputBuffers(num_reqs, num_reqs, torch.device(args.device))
        draft_tokens = torch.randint(0, 32_000, (num_reqs, 2), device=args.device)
        seq_lens = torch.randint(8, 4096, (num_reqs,), device=args.device, dtype=torch.int32)
        rejected = torch.randint(0, 4, (num_reqs,), device=args.device, dtype=torch.int32)
        buffers.positions[:num_reqs] = torch.randint(0, 4096, (num_reqs,), device=args.device)
        fn = lambda: prepare_decode_inputs(draft_tokens, seq_lens, rejected, buffers,
                                           4096, num_reqs)
        latency_us = bench(fn, args.warmup, args.repeat)
        print(f"op=_prepare_decode_inputs_kernel num_reqs={num_reqs} "
              f"latency_us={latency_us:.2f} checksum={int(buffers.input_ids.sum().item())}")


if __name__ == "__main__":
    main()
