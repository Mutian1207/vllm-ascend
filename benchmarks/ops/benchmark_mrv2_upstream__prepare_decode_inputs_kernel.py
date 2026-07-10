# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

import torch

from mrv2_upstream_bench_utils import bench_npu, init_triton_ascend_device_properties, set_npu_device
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import prepare_decode_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for num_reqs in [16, 128, 512]:
        buffers = InputBuffers(num_reqs, num_reqs, torch.device(args.device))
        draft_tokens = torch.randint(0, 32_000, (num_reqs, 2), device=args.device)
        seq_lens = torch.randint(8, 4096, (num_reqs,), device=args.device, dtype=torch.int32)
        rejected = torch.randint(0, 4, (num_reqs,), device=args.device, dtype=torch.int32)
        buffers.positions[:num_reqs] = torch.randint(0, 4096, (num_reqs,), device=args.device)
        fn = lambda: prepare_decode_inputs(draft_tokens, seq_lens, rejected, buffers,
                                           4096, num_reqs)
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_prepare_decode_inputs_kernel num_reqs={num_reqs} "
              f"latency_us={latency_us:.2f} checksum={int(buffers.input_ids.sum().item())}")


if __name__ == "__main__":
    main()

