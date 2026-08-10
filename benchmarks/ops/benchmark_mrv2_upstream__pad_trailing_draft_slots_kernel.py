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
    _pad_trailing_draft_slots_kernel,
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
    num_groups = 4
    num_speculative_steps = 3
    num_tokens_padded = 256
    query_start_loc = torch.arange(
        0, (num_reqs + 1) * (num_speculative_steps + 1),
        num_speculative_steps + 1, device=args.device, dtype=torch.int32
    )
    last_token_indices = (query_start_loc[1:] - 1).clone()
    slot_mappings = torch.zeros(
        (num_groups, num_tokens_padded), device=args.device, dtype=torch.int32
    )
    for pad_id in [-1, 0]:
        fn = lambda: _pad_trailing_draft_slots_kernel[(num_groups, num_reqs)](
            slot_mappings,
            slot_mappings.stride(0),
            query_start_loc,
            last_token_indices,
            pad_id,
            BLOCK_SIZE=256,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_pad_trailing_draft_slots_kernel num_groups={num_groups} "
            f"num_reqs={num_reqs} pad_id={pad_id} latency_us={latency_us:.2f} "
            f"checksum={int(slot_mappings.sum().item())}"
        )


if __name__ == "__main__":
    main()
