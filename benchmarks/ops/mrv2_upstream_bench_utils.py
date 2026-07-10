# SPDX-License-Identifier: Apache-2.0
"""Utilities for benchmarking upstream MRv2 Triton ops on NPU."""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, TypeVar

import torch
import torch_npu  # noqa: F401


T = TypeVar("T")

# These benchmarks import upstream vLLM kernel modules directly. Avoid loading
# the vLLM Ascend plugin while vLLM itself is being imported.
os.environ.setdefault("VLLM_PLUGINS", "")


def init_triton_ascend_device_properties() -> None:
    module = sys.modules.get("vllm_ascend.ops.triton.triton_utils")
    if module is None:
        from vllm_ascend.ops.triton import triton_utils as module
    module.init_device_properties_triton()


def set_npu_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type != "npu":
        raise ValueError(f"Expected an NPU device, got {device}")
    torch.npu.set_device(device)
    return device


def bench_npu(fn: Callable[[], T], warmup: int, repeat: int) -> tuple[float, T | None]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat, out
