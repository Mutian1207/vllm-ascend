# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from mrv2_upstream_bench_utils import (
    bench_npu,
    init_triton_ascend_device_properties,
    set_npu_device,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    prepare_prefill_inputs as ar_prepare_prefill_inputs,
)


def case_ar_prepare_prefill_inputs(device):
    for num_reqs, qlen in ((16, 8), (64, 16)):
        max_num_reqs = max(128, num_reqs)
        total = num_reqs * qlen
        buffers = InputBuffers(max_num_reqs, total, torch.device(device))
        last_idx = torch.empty(max_num_reqs, device=device, dtype=torch.int64)
        step = torch.zeros((), device=device, dtype=torch.int32)
        input_batch = SimpleNamespace(num_reqs=num_reqs, input_ids=torch.randint(0, 32000, (total,), device=device, dtype=torch.int32), positions=torch.arange(total, device=device, dtype=torch.int64), idx_mapping=torch.arange(num_reqs, device=device, dtype=torch.int32), query_start_loc=torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * qlen, seq_lens=torch.full((num_reqs,), qlen, device=device, dtype=torch.int32))
        num_sampled = torch.ones(num_reqs, device=device, dtype=torch.int32)
        num_rejected = torch.zeros(num_reqs, device=device, dtype=torch.int32)
        last = torch.randint(0, 32000, (max_num_reqs,), device=device, dtype=torch.int64)
        next_prefill = torch.randint(0, 32000, (max_num_reqs,), device=device, dtype=torch.int32)
        yield f"num_reqs={num_reqs} query_len={qlen}", lambda: ar_prepare_prefill_inputs(last_idx, step, buffers, input_batch, num_sampled, num_rejected, last, next_prefill, max_num_reqs), lambda: int(last_idx[:num_reqs].sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    set_npu_device(args.device)
    init_triton_ascend_device_properties()

    for spec, fn, checksum in case_ar_prepare_prefill_inputs(args.device):
        latency_us, _ = bench_npu(fn, args.warmup, args.repeat)
        print(f"op=_ar_prepare_prefill_inputs_kernel {spec} latency_us={latency_us:.2f} "
              f"checksum={checksum():.3f}")


if __name__ == "__main__":
    main()
