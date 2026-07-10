# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _rejection_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    init_triton_ascend_device_properties()

    for num_reqs, steps, vocab_size in [(16, 1, 32_000), (64, 3, 32_000)]:
        num_logits = num_reqs * (steps + 1)
        num_blocks = triton.cdiv(vocab_size, 8192)
        padded_blocks = triton.next_power_of_2(num_blocks)
        sampled = torch.empty((num_reqs, steps + 1), device=args.device, dtype=torch.int64)
        num_sampled = torch.empty(num_reqs, device=args.device, dtype=torch.int32)
        target_lse = torch.zeros(num_reqs, device=args.device)
        draft_lse = torch.zeros(num_reqs, device=args.device)
        target_logits = torch.randn((num_logits, vocab_size), device=args.device)
        target_argmax = torch.randint(0, vocab_size, (num_logits, num_blocks),
                                      device=args.device, dtype=torch.int64)
        target_max = torch.randn((num_logits, num_blocks), device=args.device)
        target_sumexp = torch.rand((num_logits, num_blocks), device=args.device) + 1
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), device=args.device)
        draft_logits = torch.randn((num_reqs, steps, vocab_size), device=args.device)
        draft_max = torch.randn((num_logits, num_blocks), device=args.device)
        draft_sumexp = torch.rand((num_logits, num_blocks), device=args.device) + 1
        cu_num_logits = torch.arange(num_reqs + 1, device=args.device,
                                     dtype=torch.int32) * (steps + 1)
        idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
        temp = torch.zeros(num_reqs, device=args.device)
        seed = torch.arange(num_reqs, device=args.device, dtype=torch.int64)
        pos = torch.arange(num_logits, device=args.device, dtype=torch.int32)
        fn = lambda: _rejection_kernel[(num_reqs,)](
            sampled, sampled.stride(0), num_sampled, target_lse, draft_lse,
            target_logits, target_logits.stride(0), target_argmax,
            target_argmax.stride(0), target_max, target_max.stride(0),
            target_sumexp, target_sumexp.stride(0), draft_sampled, draft_logits,
            draft_logits.stride(0), draft_logits.stride(1), draft_max,
            draft_max.stride(0), draft_sumexp, draft_sumexp.stride(0), cu_num_logits,
            idx_mapping, temp, seed, pos, None, num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_blocks, HAS_DRAFT_LOGITS=True,
            SYNTHETIC_MODE=False, num_warps=1)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_rejection_kernel num_reqs={num_reqs} steps={steps} "
              f"latency_us={latency_us:.2f} checksum={int(num_sampled.sum().item())}")


if __name__ == "__main__":
    main()

