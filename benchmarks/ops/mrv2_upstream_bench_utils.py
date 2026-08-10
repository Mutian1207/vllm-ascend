# SPDX-License-Identifier: Apache-2.0
"""Utilities for benchmarking upstream MRv2 Triton ops on NPU."""

from __future__ import annotations

import os
import sys
import time
import types
from typing import Callable, TypeVar

# Workaround for a broken/mixed triton install on this machine: the
# `triton.experimental.gluon` submodule present on disk is the NVIDIA upstream
# (GPU) variant, but it is incompatible with the installed triton-ascend base
# (it references newer-triton symbols such as `constexpr_type` /
# `constexpr_function` that do not exist here). vLLM imports gluon
# unconditionally, so every benchmark would die at
# `from vllm.triton_utils import triton`. None of the upstream MRv2 kernels
# benchmarked here use Gluon IR (they are plain @triton.jit kernels), so we
# satisfy vLLM's import with lightweight stub modules. We only inject the stub
# if the real gluon fails to import, so a healthy triton install keeps using
# its own gluon. sys.modules caching makes Python skip the broken package.
try:
    import triton.experimental.gluon  # noqa: F401  (try the real one first)
except Exception:  # noqa: BLE001 - any failure means fall back to the stub
    _gluon = types.ModuleType("triton.experimental.gluon")
    _gluon_language = types.ModuleType("triton.experimental.gluon.language")
    _gluon.language = _gluon_language
    sys.modules["triton.experimental.gluon"] = _gluon
    sys.modules["triton.experimental.gluon.language"] = _gluon_language

# vLLM's triton_utils re-exports a handful of helper symbols. In a broken or
# mixed environment (e.g. a stale __pycache__, or a vLLM checkout whose
# triton-ascend base predates them) some of these may be absent, which would
# abort every benchmark at import time. None of the upstream MRv2 kernels
# benchmarked here use them, so we backfill the safe ones. `triton`/`tl`/
# `tldevice` are real modules and must exist; only the helper symbols below are
# stubbed:
#   - `_aggregate`        (re-exported from triton.language.core)
#   - `use_tensor_descriptor` (re-exported from vllm.triton_utils.tensor_descriptor;
#                             resolves to False on Ascend, which is correct)
try:
    import triton.language.core as _triton_lang_core
    if not hasattr(_triton_lang_core, "_aggregate"):
        _triton_lang_core._aggregate = lambda *args, **kwargs: None  # noqa: E731
except ImportError:
    pass

try:
    import vllm.triton_utils as _vllm_triton_utils

    def _use_tensor_descriptor(override: bool | None = None) -> bool:
        return False

    for _name, _fb in (
        ("use_tensor_descriptor", _use_tensor_descriptor),
    ):
        if not hasattr(_vllm_triton_utils, _name):
            setattr(_vllm_triton_utils, _name, _fb)
except ImportError:
    pass

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None  # CUDA-only environments have no torch_npu


T = TypeVar("T")

# These benchmarks import upstream vLLM kernel modules directly. Avoid loading
# the vLLM Ascend plugin while vLLM itself is being imported.
os.environ.setdefault("VLLM_PLUGINS", "")


def init_triton_ascend_device_properties(device: str | torch.device | None = None) -> None:
    # Ascend-specific device-property init (num_aicore / num_vectorcore). On a
    # CUDA device this is irrelevant, so skip it. Importing
    # vllm_ascend.ops.triton.triton_utils forces the parent vllm_ascend.ops
    # package to initialize, which (in a version-mismatched environment) can
    # fail on imports like `from vllm.model_executor.layers.fused_moe.layer
    # import FusedMoE` because this vLLM has dropped the standalone FusedMoE
    # class. The upstream MRv2 kernels benchmarked here are plain @triton.jit
    # kernels and do not depend on those properties, so skip gracefully.
    if device is not None and torch.device(device).type != "npu":
        return
    if torch_npu is None:
        return  # no Ascend backend in this environment (e.g. pure CUDA)
    try:
        module = sys.modules.get("vllm_ascend.ops.triton.triton_utils")
        if module is None:
            from vllm_ascend.ops.triton import triton_utils as module
    except ImportError as exc:
        print(
            f"[warn] skipping triton-ascend device init "
            f"(vllm-ascend appears version-mismatched with vLLM): {exc}"
        )
        return
    module.init_device_properties_triton()


def set_npu_device(device: str | torch.device) -> torch.device:
    """Set the active device. Works for both ``npu`` and ``cuda`` backends."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("torch_npu is not available in this environment")
        torch.npu.set_device(device)
    else:
        raise ValueError(f"Unsupported device type: {device}")
    return device


def _synchronize() -> None:
    """Synchronize every backend that is actually available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch_npu is not None and torch.npu.is_available():
        torch.npu.synchronize()


def bench_npu(fn: Callable[[], T], warmup: int, repeat: int) -> tuple[float, T | None]:
    """Benchmark ``fn`` and return (latency_us, last_output).

    Synchronizes whichever backend (CUDA and/or NPU) is present, so the same
    helper works on both.
    """
    out = None
    for _ in range(warmup):
        out = fn()
    _synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    _synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat, out
