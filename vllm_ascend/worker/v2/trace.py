# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace logs for understanding the Ascend MRv2 execution flow."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from vllm.logger import logger


def log(stage: str, message: str, *args: Any) -> None:
    logger.info("[MRv2 trace][%s] " + message, stage, *args)


def describe_value(value: Any, *, max_items: int = 6) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": value.numel(),
            "data_ptr": hex(value.data_ptr()) if value.device.type != "meta" else None,
        }
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return {
            "type": "ndarray",
            "shape": value.shape,
            "dtype": str(value.dtype),
            "sample": flat[:max_items].tolist(),
        }
    if isinstance(value, Mapping):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "sample": list(value.items())[:max_items],
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "sample": list(value[:max_items]),
        }
    return value


def describe_attrs(obj: Any, attrs: Sequence[str], *, max_items: int = 6) -> dict[str, Any]:
    return {
        attr: describe_value(getattr(obj, attr, None), max_items=max_items)
        for attr in attrs
    }
