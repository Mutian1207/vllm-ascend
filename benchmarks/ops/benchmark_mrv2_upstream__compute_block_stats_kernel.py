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
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_block_stats_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for num_reqs, steps, vocab_size, temp_value in [(16, 3, 32_000, 0.0),
                                                    (16, 3, 32_000, 0.7)]:
        num_logits = num_reqs * (steps + 1)
        block_size = 8192
        num_blocks = triton.cdiv(vocab_size, block_size)
        target_logits = torch.randn((num_logits, vocab_size),
                                    device=args.device)
        draft_logits = torch.randn((num_reqs, steps, vocab_size),
                                   device=args.device)
        shape = (num_logits, num_blocks)
        target_argmax = torch.empty(shape,
                                    device=args.device,
                                    dtype=torch.int64)
        target_max = torch.empty(shape, device=args.device)
        target_sumexp = torch.empty(shape, device=args.device)
        draft_max = torch.empty(shape, device=args.device)
        draft_sumexp = torch.empty(shape, device=args.device)
        expanded_idx = torch.arange(num_reqs,
                                    device=args.device,
                                    dtype=torch.int32).repeat_interleave(steps + 1)
        expanded_pos = torch.arange(steps + 1,
                                    device=args.device,
                                    dtype=torch.int32).repeat(num_reqs)
        temp = torch.full((num_reqs,), temp_value, device=args.device)
        fn = lambda: _compute_block_stats_kernel[(num_logits, num_blocks)](
            target_argmax, target_argmax.stride(0), target_max,
            target_max.stride(0), target_sumexp, target_sumexp.stride(0),
            draft_max, draft_max.stride(0), draft_sumexp,
            draft_sumexp.stride(0), target_logits, target_logits.stride(0),
            draft_logits, draft_logits.stride(0), draft_logits.stride(1),
            expanded_idx, expanded_pos, temp, vocab_size, steps,
            BLOCK_SIZE=block_size, HAS_DRAFT_LOGITS=True)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_compute_block_stats_kernel num_logits={num_logits} "
              f"vocab_size={vocab_size} temp={temp_value} "
              f"latency_us={latency_us:.2f} "
              f"checksum={float(target_max.sum().item()):.3f}")


if __name__ == "__main__":
    main()
