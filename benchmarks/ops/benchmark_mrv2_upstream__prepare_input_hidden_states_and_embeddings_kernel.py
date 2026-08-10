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
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    _prepare_input_hidden_states_and_embeddings_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    max_num_reqs = 16
    num_speculative_steps = 3
    hidden_size = 1024
    max_query_len = 256
    query_block_size = 16
    hidden_block_size = 256
    for num_reqs, max_query_len_loop in [(16, 64), (16, 256)]:
        query_start_loc = torch.arange(
            0, (num_reqs + 1) * (num_speculative_steps + 1),
            num_speculative_steps + 1, device=args.device, dtype=torch.int32
        )
        num_rejected = torch.ones(num_reqs, device=args.device, dtype=torch.int32)
        idx_mapping = torch.arange(num_reqs, device=args.device, dtype=torch.int32)
        hidden_states = torch.randn(
            (max_query_len, hidden_size), device=args.device
        )
        target_hidden_states = torch.randn(
            (max_query_len, hidden_size), device=args.device
        )
        cached_target_hidden_states = torch.randn(
            (max_num_reqs, num_speculative_steps - 1, hidden_size), device=args.device
        )
        input_embeds = torch.randn(
            (max_query_len, hidden_size), device=args.device
        )
        cached_draft_input_embeds = torch.randn(
            (max_num_reqs, num_speculative_steps - 1, hidden_size), device=args.device
        )
        grid = (
            num_reqs,
            triton.cdiv(max_query_len_loop, query_block_size),
            triton.cdiv(hidden_size, hidden_block_size),
        )
        fn = lambda: _prepare_input_hidden_states_and_embeddings_kernel[grid](
            hidden_states,
            hidden_states.stride(0),
            target_hidden_states,
            target_hidden_states.stride(0),
            cached_target_hidden_states,
            cached_target_hidden_states.stride(0),
            cached_target_hidden_states.stride(1),
            input_embeds,
            input_embeds.stride(0),
            cached_draft_input_embeds,
            cached_draft_input_embeds.stride(0),
            cached_draft_input_embeds.stride(1),
            idx_mapping,
            num_rejected,
            query_start_loc,
            num_speculative_steps,
            hidden_size,
            BLOCK_SIZE_Q=query_block_size,
            BLOCK_SIZE_H=hidden_block_size,
            USE_INPUT_EMBEDS=True,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_prepare_input_hidden_states_and_embeddings_kernel "
            f"num_reqs={num_reqs} max_query_len={max_query_len_loop} "
            f"hidden_size={hidden_size} latency_us={latency_us:.2f} "
            f"checksum={float(hidden_states.sum().item()):.3f}"
        )


if __name__ == "__main__":
    main()
