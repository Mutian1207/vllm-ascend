# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _resample_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    init_triton_ascend_device_properties()

    for num_reqs, steps, vocab_size in [(16, 1, 32_000), (64, 3, 32_000)]:
        block_size = 1024
        num_blocks = triton.cdiv(vocab_size, block_size)
        num_logits = num_reqs * (steps + 1)
        out_idx = torch.empty((num_reqs, num_blocks), device=args.device, dtype=torch.int64)
        out_max = torch.empty((num_reqs, num_blocks), device=args.device)
        target_logits = torch.randn((num_logits, vocab_size), device=args.device)
        target_lse = torch.zeros(num_reqs, device=args.device)
        draft_logits = torch.randn((num_reqs, steps, vocab_size), device=args.device)
        draft_lse = torch.zeros(num_reqs, device=args.device)
        rejected_step = torch.zeros(num_reqs, device=args.device, dtype=torch.int32)
        cu_num_logits = torch.arange(num_reqs + 1, device=args.device,
                                     dtype=torch.int32) * (steps + 1)
        expanded_idx = torch.arange(num_reqs, device=args.device,
                                    dtype=torch.int32).repeat_interleave(steps + 1)
        draft_sampled = torch.randint(0, vocab_size, (num_logits,), device=args.device)
        temp = torch.ones(num_reqs, device=args.device)
        seed = torch.arange(num_reqs, device=args.device, dtype=torch.int64)
        pos = torch.arange(num_logits, device=args.device, dtype=torch.int32)
        fn = lambda: _resample_kernel[(num_reqs, num_blocks)](
            out_idx, out_idx.stride(0), out_max, out_max.stride(0), target_logits,
            target_logits.stride(0), target_lse, draft_logits, draft_logits.stride(0),
            draft_logits.stride(1), draft_lse, rejected_step, cu_num_logits,
            expanded_idx, draft_sampled, temp, seed, pos, vocab_size,
            BLOCK_SIZE=block_size, HAS_DRAFT_LOGITS=True, USE_FP64=False)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_resample_kernel num_reqs={num_reqs} steps={steps} "
              f"vocab_size={vocab_size} latency_us={latency_us:.2f} "
              f"checksum={int(out_idx.sum().item())}")


if __name__ == "__main__":
    main()

