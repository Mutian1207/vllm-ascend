# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    idx_mapping_stride,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_rows,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (num_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, num_rows)

    for row_idx in range(start_row, end_row):
        req_state_idx = tl.load(idx_mapping_ptr + row_idx * idx_mapping_stride)
        total_len = tl.load(total_len_ptr + req_state_idx)
        num_sampled = tl.load(num_sampled_ptr + row_idx)

        if num_sampled > 0:
            token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + num_sampled - 1)
            tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
            tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

        for i in range(num_sampled):
            token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + i)

            token_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + token_id
            count = tl.load(token_ptr)
            count += 1
            tl.store(token_ptr, count)

            tl.store(
                all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
                token_id,
            )

        query_start = tl.load(query_start_loc_ptr + row_idx)
        query_end = tl.load(query_start_loc_ptr + row_idx + 1)
        query_len = query_end - query_start
        num_rejected = tl.load(num_rejected_ptr + row_idx)

        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        num_computed += query_len - num_rejected
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)


def post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None:
    num_rows = idx_mapping.shape[0]
    grid = (min(num_rows, get_vectorcore_num()),)
    _post_update_kernel[grid](
        idx_mapping,
        idx_mapping.stride(0),
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0),
        sampled_tokens,
        sampled_tokens.stride(0),
        num_rows,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
    )


def case_post_update(device):
    for num_reqs, steps in ((16, 2), (128, 4)):
        max_len = 4096
        idx = torch.arange(num_reqs, device=device, dtype=torch.int32)
        num_comp = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        last = torch.zeros(num_reqs, device=device, dtype=torch.int64)
        output_bin_counts = torch.zeros((num_reqs, 32000), device=device, dtype=torch.int32)
        sampled = torch.randint(0, 32000, (num_reqs, steps + 1), device=device, dtype=torch.int64)
        num_sampled = torch.full((num_reqs,), steps + 1, device=device, dtype=torch.int32)
        num_rejected = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        qsl = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * (steps + 1)
        all_ids = torch.zeros((num_reqs, max_len), device=device, dtype=torch.int32)
        total_len = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        yield f"num_reqs={num_reqs} steps={steps}", lambda: post_update(idx, num_comp, last, output_bin_counts, sampled, num_sampled, num_rejected, qsl, all_ids, total_len), lambda: int(total_len.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_post_update(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_post_update_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
