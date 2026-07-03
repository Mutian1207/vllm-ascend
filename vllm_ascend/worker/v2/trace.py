# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in trace logs for understanding the Ascend MRv2 execution flow."""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TRACE_MRV2", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def log(stage: str, message: str, *args: Any) -> None:
    if enabled():
        logger.info("[MRv2 trace][%s] " + message, stage, *args)
