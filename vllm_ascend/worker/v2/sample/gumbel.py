# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import os

import torch
from vllm.triton_utils import tl, triton


_GUMBEL_INPUT_DUMP_COUNT = 0


def _cpu_or_none(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _dump_gumbel_inputs(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    sampled: torch.Tensor,
    output_processed_logits: torch.Tensor | None,
    output_processed_logits_col: torch.Tensor | None,
) -> None:
    global _GUMBEL_INPUT_DUMP_COUNT

    dump_dir = os.getenv("VLLM_ASCEND_DUMP_GUMBEL_INPUTS")
    if not dump_dir:
        return

    limit = int(os.getenv("VLLM_ASCEND_DUMP_GUMBEL_LIMIT", "20"))
    if _GUMBEL_INPUT_DUMP_COUNT >= limit:
        return

    os.makedirs(dump_dir, exist_ok=True)
    call_idx = _GUMBEL_INPUT_DUMP_COUNT
    _GUMBEL_INPUT_DUMP_COUNT += 1

    metadata = {
        "pid": os.getpid(),
        "call_idx": call_idx,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_stride": list(logits.stride()),
        "expanded_idx_mapping_shape": list(expanded_idx_mapping.shape),
        "expanded_idx_mapping_dtype": str(expanded_idx_mapping.dtype),
        "temperature_shape": list(temperature.shape),
        "temperature_dtype": str(temperature.dtype),
        "seed_shape": list(seed.shape),
        "seed_dtype": str(seed.dtype),
        "pos_shape": list(pos.shape),
        "pos_dtype": str(pos.dtype),
        "apply_temperature": bool(apply_temperature),
        "sampled_shape": list(sampled.shape),
        "sampled_dtype": str(sampled.dtype),
        "has_output_processed_logits": output_processed_logits is not None,
        "output_processed_logits_shape": (
            list(output_processed_logits.shape)
            if output_processed_logits is not None
            else None
        ),
        "output_processed_logits_dtype": (
            str(output_processed_logits.dtype)
            if output_processed_logits is not None
            else None
        ),
        "has_output_processed_logits_col": output_processed_logits_col is not None,
        "output_processed_logits_col_shape": (
            list(output_processed_logits_col.shape)
            if output_processed_logits_col is not None
            else None
        ),
        "output_processed_logits_col_dtype": (
            str(output_processed_logits_col.dtype)
            if output_processed_logits_col is not None
            else None
        ),
    }

    case = {
        "metadata": metadata,
        "logits": logits.detach().cpu(),
        "expanded_idx_mapping": expanded_idx_mapping.detach().cpu(),
        "temperature": temperature.detach().cpu(),
        "seed": seed.detach().cpu(),
        "pos": pos.detach().cpu(),
        "apply_temperature": bool(apply_temperature),
        "sampled_ascend": sampled.detach().cpu(),
        "output_processed_logits": _cpu_or_none(output_processed_logits),
        "output_processed_logits_col": _cpu_or_none(output_processed_logits_col),
    }
    filename = f"gumbel_case_pid{os.getpid()}_{call_idx:05d}.pt"
    torch.save(case, os.path.join(dump_dir, filename))


@triton.jit(do_not_specialize=["logits_stride", "vocab_size"])
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """
    Args:
        logits: Tensor of shape (num_tokens, vocab_size) containing the logits.
        expanded_idx_mapping: Tensor containing the mapping from token index
            to request index of tensor temperature.
        temperature: Tensor containing the temperature value for each request.
    """
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 44032
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        multibuffer=False,
    )


@triton.jit(
    do_not_specialize=[
        "local_argmax_stride",
        "local_max_stride",
        "processed_logits_stride",
        "logits_stride",
        "vocab_size",
    ]
)
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    if temp != 0.0 and APPLY_TEMPERATURE:
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        logits = logits / temp

    if processed_logits_ptr is not None:
        # Store the temperature-applied logits.
        if processed_logits_col_ptr is not None:
            col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr + req_state_idx * processed_logits_stride + col * vocab_size + block,
            logits,
            mask=mask,
        )

    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        # NOTE(Ronald1995): change pos's dtype to tl.int32, because triton-ascend's
        # compiler doesn't support uint64 of pos arg.
        pos = tl.load(pos_ptr + token_idx).to(tl.int32)
        gumbel_seed = tl.randint(seed, pos)

        # NOTE(Ronald1995): r is tl.float64 in vllm, change it to tl.float32,
        # because triton-ascend's compiler does not support float64.
        r = tl.rand(gumbel_seed, block).to(tl.float32)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    idx = tl.argmax(logits, axis=0)
    token_id = block_idx * BLOCK_SIZE + idx
    value = tl.max(logits, axis=0)
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    if use_fp64:
        raise NotImplementedError("FP64 Gumbel sampling is not supported on NPU.")
    logits = logits.to(torch.float32)
    sampled = torch.ops._C_ascend.npu_gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        output_processed_logits,
        output_processed_logits_col,
    )
    _dump_gumbel_inputs(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        sampled,
        output_processed_logits,
        output_processed_logits_col,
    )
    return sampled
