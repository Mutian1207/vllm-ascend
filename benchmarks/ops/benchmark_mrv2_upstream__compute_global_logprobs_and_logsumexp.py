# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logprobs_and_logsumexp,
)


@triton.jit
def _bench_kernel(out_target, out_draft, out_target_lse, out_draft_lse, token,
                  req_state_idx, draft_step, target_logits, target_logits_stride,
                  target_local_max, target_local_max_stride, target_local_sumexp,
                  target_local_sumexp_stride, draft_logits, draft_logits_stride_0,
                  draft_logits_stride_1, draft_local_max, draft_local_max_stride,
                  draft_local_sumexp, draft_local_sumexp_stride, vocab_num_blocks,
                  PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
                  HAS_DRAFT_LOGITS: tl.constexpr):
    row = tl.program_id(0)
    mask = token < target_logits_stride
    t, d, tlse, dlse = _compute_global_logprobs_and_logsumexp(
        token, mask, row, req_state_idx, draft_step, target_logits,
        target_logits_stride, target_local_max, target_local_max_stride,
        target_local_sumexp, target_local_sumexp_stride, draft_logits,
        draft_logits_stride_0, draft_logits_stride_1, draft_local_max,
        draft_local_max_stride, draft_local_sumexp, draft_local_sumexp_stride,
        vocab_num_blocks, PADDED_VOCAB_NUM_BLOCKS, HAS_DRAFT_LOGITS)
    tl.store(out_target + row, t)
    tl.store(out_draft + row, d)
    tl.store(out_target_lse + row, tlse)
    tl.store(out_draft_lse + row, dlse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for num_logits, num_blocks, has_draft_logits in [(64, 4, True), (64, 19, False)]:
        num_reqs = 16
        steps = 3
        vocab_size = num_blocks * 8192
        padded = triton.next_power_of_2(num_blocks)
        target_logits = torch.randn((num_logits, vocab_size), device=args.device)
        draft_logits = torch.randn((num_reqs, steps, vocab_size), device=args.device)
        target_local_max = torch.randn((num_logits, num_blocks), device=args.device)
        target_local_sumexp = torch.rand((num_logits, num_blocks), device=args.device) + 1
        draft_local_max = torch.randn((num_logits, num_blocks), device=args.device)
        draft_local_sumexp = torch.rand((num_logits, num_blocks), device=args.device) + 1
        out_target = torch.empty(num_logits, device=args.device)
        out_draft = torch.empty(num_logits, device=args.device)
        out_target_lse = torch.empty(num_logits, device=args.device)
        out_draft_lse = torch.empty(num_logits, device=args.device)
        fn = lambda: _bench_kernel[(num_logits,)](
            out_target, out_draft, out_target_lse, out_draft_lse, 17, 0, 1,
            target_logits, target_logits.stride(0), target_local_max,
            target_local_max.stride(0), target_local_sumexp,
            target_local_sumexp.stride(0), draft_logits, draft_logits.stride(0),
            draft_logits.stride(1), draft_local_max, draft_local_max.stride(0),
            draft_local_sumexp, draft_local_sumexp.stride(0), num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded, HAS_DRAFT_LOGITS=has_draft_logits)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        checksum = float(out_target.sum().item()) + float(out_draft.sum().item())
        print(f"op=_compute_global_logprobs_and_logsumexp num_logits={num_logits} "
              f"num_blocks={num_blocks} has_draft_logits={has_draft_logits} "
              f"latency_us={latency_us:.2f} checksum={checksum:.3f}")


if __name__ == "__main__":
    main()
