# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import update_draft_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    init_triton_ascend_device_properties()

    for num_reqs, hidden_size in [(16, 4096), (128, 4096), (128, 8192)]:
        buffers = InputBuffers(num_reqs, num_reqs, torch.device(args.device))
        draft_tokens = torch.randint(0, 32_000, (num_reqs,), device=args.device)
        step = torch.tensor(1, device=args.device, dtype=torch.int64)
        hidden_states = torch.randn((num_reqs, hidden_size), device=args.device)
        output_draft_tokens = torch.empty((num_reqs, 4), device=args.device, dtype=torch.int64)
        next_hidden = torch.empty_like(hidden_states)
        buffers.positions[:num_reqs] = torch.arange(num_reqs, device=args.device)
        buffers.seq_lens[:num_reqs] = torch.arange(1, num_reqs + 1, device=args.device,
                                                   dtype=torch.int32)
        fn = lambda: update_draft_inputs(draft_tokens, step, hidden_states,
                                         output_draft_tokens, next_hidden, buffers,
                                         num_reqs, 4096, 4)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_update_draft_inputs_kernel num_reqs={num_reqs} hidden_size={hidden_size} "
              f"latency_us={latency_us:.2f} checksum={int(output_draft_tokens[:, 1].sum().item())}")


if __name__ == "__main__":
    main()

