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
    _cache_inputs_kernel,
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
    hidden_block_size = 1024
    query_start_loc = torch.arange(
        0, (max_num_reqs + 1) * (num_speculative_steps + 1),
        num_speculative_steps + 1, device=args.device, dtype=torch.int32
    )
    last_token_indices = (query_start_loc[1:] - 1).clone()
    idx_mapping = torch.arange(max_num_reqs, device=args.device, dtype=torch.int32)
    draft_input_ids = torch.randint(
        0, 32000, (max_query_len,), device=args.device, dtype=torch.int32
    )
    draft_input_embeds = torch.randn(
        (max_query_len, hidden_size), device=args.device
    )
    draft_input_hidden_states = torch.randn(
        (max_query_len, hidden_size), device=args.device
    )
    cached_draft_input_ids = torch.randint(
        0, 32000, (max_num_reqs, num_speculative_steps - 1),
        device=args.device, dtype=torch.int32
    )
    cached_draft_input_embeds = torch.randn(
        (max_num_reqs, num_speculative_steps - 1, hidden_size), device=args.device
    )
    cached_target_hidden_states = torch.randn(
        (max_num_reqs, num_speculative_steps - 1, hidden_size), device=args.device
    )
    for num_reqs in [1, 16]:
        fn = lambda: _cache_inputs_kernel[
            (num_reqs, triton.cdiv(hidden_size, hidden_block_size))
        ](
            draft_input_ids,
            draft_input_embeds,
            draft_input_embeds.stride(0),
            draft_input_hidden_states,
            draft_input_hidden_states.stride(0),
            cached_draft_input_ids,
            cached_draft_input_ids.stride(0),
            cached_draft_input_embeds,
            cached_draft_input_embeds.stride(0),
            cached_draft_input_embeds.stride(1),
            cached_target_hidden_states,
            cached_target_hidden_states.stride(0),
            cached_target_hidden_states.stride(1),
            idx_mapping,
            last_token_indices,
            query_start_loc,
            num_speculative_steps,
            hidden_size,
            BLOCK_SIZE=hidden_block_size,
            USE_INPUT_EMBEDS=True,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_cache_inputs_kernel num_reqs={num_reqs} hidden_size={hidden_size} "
            f"latency_us={latency_us:.2f} "
            f"checksum={float(cached_target_hidden_states.sum().item()):.3f}"
        )


if __name__ == "__main__":
    main()
