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
    _compute_cumulative_log_p_kernel,
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
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)

    target_logits = torch.randn((num_logits, vocab_size), device=args.device)
    draft_logits = torch.randn(
        (max_num_reqs, num_speculative_steps, vocab_size), device=args.device
    )
    draft_sampled = torch.randint(
        0, vocab_size, (num_logits,), device=args.device, dtype=torch.int64
    )
    cu_num_logits = torch.arange(
        0, (num_reqs + 1) * (num_speculative_steps + 1),
        num_speculative_steps + 1, device=args.device, dtype=torch.int32
    )
    idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
    temperature = torch.full(
        (max_num_reqs,), 0.7, device=args.device, dtype=torch.float32
    )
    # Precomputed per-vocab-block stats (from _compute_local_logits_stats_kernel).
    target_local_max = torch.randn(
        (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
    )
    target_local_sumexp = torch.randn(
        (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
    )
    draft_local_max = torch.randn(
        (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
    )
    draft_local_sumexp = torch.randn(
        (num_logits, vocab_num_blocks), device=args.device, dtype=torch.float32
    )
    cumulative_log_p = torch.empty(num_logits, device=args.device, dtype=torch.float32)

    fn = lambda: _compute_cumulative_log_p_kernel[(num_reqs,)](
        cumulative_log_p,
        target_logits,
        target_logits.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=True,
        num_warps=1,
    )
    latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
    print(
        f"op=_compute_cumulative_log_p_kernel num_reqs={num_reqs} "
        f"vocab_size={vocab_size} latency_us={latency_us:.2f} "
        f"checksum={float(cumulative_log_p.sum().item()):.3f}"
    )


if __name__ == "__main__":
    main()
