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
    _compute_local_logits_stats_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    num_reqs = 16
    num_speculative_steps = 3
    vocab_size = 32768
    vocab_block_size = 8192
    max_num_reqs = num_reqs
    num_logits = num_reqs * (num_speculative_steps + 1)
    vocab_num_blocks = triton.cdiv(vocab_size, vocab_block_size)

    target_logits = torch.randn((num_logits, vocab_size), device=args.device)
    draft_logits = torch.randn(
        (max_num_reqs, num_speculative_steps, vocab_size), device=args.device
    )
    expanded_idx_mapping = (
        torch.arange(num_logits, device=args.device, dtype=torch.int32)
        // (num_speculative_steps + 1)
    )
    expanded_local_pos = (
        torch.arange(num_logits, device=args.device, dtype=torch.int32)
        % (num_speculative_steps + 1)
    )
    temperature = torch.full(
        (max_num_reqs,), 0.7, device=args.device, dtype=torch.float32
    )

    for has_draft in (False, True):
        target_local_argmax = torch.empty(
            (num_logits, vocab_num_blocks), device=args.device, dtype=torch.int64
        )
        target_local_max = torch.empty(
            (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
        )
        target_local_sumexp = torch.empty(
            (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
        )
        draft_local_max = torch.empty(
            (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
        )
        draft_local_sumexp = torch.empty(
            (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
        )
        dlogits = draft_logits if has_draft else None
        d_stride0 = draft_logits.stride(0) if has_draft else 0
        d_stride1 = draft_logits.stride(1) if has_draft else 0
        fn = lambda: _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            target_logits,
            target_logits.stride(0),
            dlogits,
            d_stride0,
            d_stride1,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_speculative_steps,
            BLOCK_SIZE=vocab_block_size,
            HAS_DRAFT_LOGITS=has_draft,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_compute_local_logits_stats_kernel num_logits={num_logits} "
            f"vocab_size={vocab_size} has_draft={has_draft} "
            f"latency_us={latency_us:.2f} "
            f"checksum={float(target_local_max.sum().item()):.3f}"
        )


if __name__ == "__main__":
    main()
