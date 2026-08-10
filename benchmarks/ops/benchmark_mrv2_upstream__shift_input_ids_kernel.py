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
    _shift_input_ids_kernel,
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
    max_query_len = 256
    query_start_loc = torch.arange(
        0, (max_num_reqs + 1) * (num_speculative_steps + 1),
        num_speculative_steps + 1, device=args.device, dtype=torch.int32
    )
    last_token_indices = (query_start_loc[1:] - 1).clone()
    idx_mapping = torch.arange(max_num_reqs, device=args.device, dtype=torch.int32)
    draft_tokens = torch.randint(
        0, 32000, (max_num_reqs,), device=args.device, dtype=torch.int32
    )
    for num_reqs in [1, 16]:
        input_ids = torch.randint(
            0, 32000, (max_query_len,), device=args.device, dtype=torch.int32
        )
        fn = lambda: _shift_input_ids_kernel[(num_reqs,)](
            input_ids,
            idx_mapping,
            query_start_loc,
            last_token_indices,
            draft_tokens,
            BLOCK_SIZE=1024,
        )
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(
            f"op=_shift_input_ids_kernel num_reqs={num_reqs} "
            f"latency_us={latency_us:.2f} "
            f"checksum={int(input_ids.sum().item())}"
        )


if __name__ == "__main__":
    main()
