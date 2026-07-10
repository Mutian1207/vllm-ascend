# SPDX-License-Identifier: Apache-2.0
"""Drive Ascend prepare_dflash_inputs with configurable tensor sizes for msprof."""

import argparse
import inspect
from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator

# Match normal plugin initialization order and avoid a device_op/ops circular import.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel_ascend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ascend prepare_dflash_inputs for msprof collection."
    )
    parser.add_argument("--num-reqs", type=int, default=16)
    parser.add_argument(
        "--scheduled-tokens",
        default=None,
        help=(
            "Comma-separated scheduled-token counts, one per active request. "
            "This permits replaying a non-uniform production batch."
        ),
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=64,
        help="Target tokens scheduled per request.",
    )
    parser.add_argument("--spec-steps", type=int, default=5)
    parser.add_argument(
        "--sample-from-anchor",
        action="store_true",
        help="Use the DSpark layout; the default is DFlash.",
    )
    parser.add_argument("--num-rejected", type=int, default=0)
    parser.add_argument(
        "--chunked-prefill",
        action="store_true",
        help="Use next_prefill_tokens as the anchor.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-reqs", type=int, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--draft-token-id", type=int, default=999)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of profiled kernel calls after one compile/warmup call.",
    )
    return parser.parse_args()


def resolve_scheduled_tokens(args: argparse.Namespace) -> np.ndarray:
    if args.scheduled_tokens is None:
        return np.full(args.num_reqs, args.context_len, dtype=np.int32)

    scheduled_tokens = np.fromstring(args.scheduled_tokens, sep=",", dtype=np.int32)
    if scheduled_tokens.size != args.num_reqs or np.any(scheduled_tokens <= 0):
        raise ValueError(
            "--scheduled-tokens must contain one positive integer per request"
        )
    return scheduled_tokens


def resolve_sizes(
    args: argparse.Namespace, scheduled_tokens: np.ndarray
) -> tuple[int, int, int]:
    if args.num_reqs <= 0 or args.context_len <= 0 or args.spec_steps <= 0:
        raise ValueError("num-reqs, context-len and spec-steps must be positive")
    if not 0 <= args.num_rejected < args.context_len:
        raise ValueError("num-rejected must be in [0, context-len)")
    if args.repeat <= 0:
        raise ValueError("repeat must be positive")

    num_query_per_req = (
        args.spec_steps if args.sample_from_anchor else args.spec_steps + 1
    )
    max_num_reqs = args.max_num_reqs or args.num_reqs
    if max_num_reqs < args.num_reqs:
        raise ValueError("max-num-reqs cannot be smaller than num-reqs")

    required_tokens = max(
        int(scheduled_tokens.sum()),
        args.num_reqs * num_query_per_req,
    )
    max_num_tokens = args.max_num_tokens or required_tokens
    if max_num_tokens < required_tokens:
        raise ValueError(f"max-num-tokens must be at least {required_tokens}")
    return num_query_per_req, max_num_reqs, max_num_tokens


def make_inputs(
    args: argparse.Namespace,
    scheduled_tokens: np.ndarray,
    num_query_per_req: int,
    max_num_reqs: int,
    max_num_tokens: int,
) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, torch.Tensor]]:
    device = torch.device(args.device)
    total_context_tokens = int(scheduled_tokens.sum())
    query_start_loc = np.concatenate(
        (np.array([0], dtype=np.int32), np.cumsum(scheduled_tokens, dtype=np.int32))
    )

    input_batch = SimpleNamespace(
        num_reqs=args.num_reqs,
        num_scheduled_tokens=scheduled_tokens,
        positions=torch.arange(total_context_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.from_numpy(query_start_loc).to(device),
        idx_mapping=torch.arange(args.num_reqs, dtype=torch.int32, device=device),
    )

    input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int32, device=device),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
    )

    max_num_blocks = (args.max_model_len + args.block_size - 1) // args.block_size
    tensors = {
        "query_slot_mapping": torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        ),
        "context_positions": torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        ),
        "context_slot_mapping": torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        ),
        "sample_indices": torch.zeros(
            max_num_reqs * args.spec_steps, dtype=torch.int64, device=device
        ),
        "sample_pos": torch.zeros(
            max_num_reqs * args.spec_steps, dtype=torch.int64, device=device
        ),
        "sample_idx_mapping": torch.zeros(
            max_num_reqs * args.spec_steps, dtype=torch.int32, device=device
        ),
        "temperature": torch.zeros(
            max_num_reqs, dtype=torch.float32, device=device
        ),
        "seeds": torch.zeros(max_num_reqs, dtype=torch.int64, device=device),
        "num_sampled": torch.full(
            (args.num_reqs,),
            0 if args.chunked_prefill else 1,
            dtype=torch.int32,
            device=device,
        ),
        "num_rejected": torch.full(
            (args.num_reqs,), args.num_rejected, dtype=torch.int32, device=device
        ),
        "last_sampled": torch.arange(
            1000, 1000 + max_num_reqs, dtype=torch.int32, device=device
        ).view(max_num_reqs, 1),
        "next_prefill_tokens": torch.arange(
            2000, 2000 + max_num_reqs, dtype=torch.int32, device=device
        ),
        "input_temperature": torch.ones(
            max_num_reqs, dtype=torch.float32, device=device
        ),
        "input_seeds": torch.arange(
            max_num_reqs, dtype=torch.int64, device=device
        ),
        # The values only need to describe valid physical blocks for this workload.
        "block_table": torch.arange(
            max_num_reqs * max_num_blocks, dtype=torch.int32, device=device
        ).view(max_num_reqs, max_num_blocks),
    }
    return input_buffers, input_batch, tensors


def make_call(
    args: argparse.Namespace,
    input_buffers: SimpleNamespace,
    input_batch: SimpleNamespace,
    tensors: dict[str, torch.Tensor],
    num_query_per_req: int,
    max_num_reqs: int,
    max_num_tokens: int,
):
    kwargs = dict(
        input_buffers=input_buffers,
        query_slot_mapping=tensors["query_slot_mapping"],
        context_positions=tensors["context_positions"],
        context_slot_mapping=tensors["context_slot_mapping"],
        sample_indices=tensors["sample_indices"],
        sample_pos=tensors["sample_pos"],
        sample_idx_mapping=tensors["sample_idx_mapping"],
        input_batch=input_batch,
        num_sampled=tensors["num_sampled"],
        num_rejected=tensors["num_rejected"],
        last_sampled=tensors["last_sampled"],
        next_prefill_tokens=tensors["next_prefill_tokens"],
        block_table=tensors["block_table"],
        block_size=args.block_size,
        parallel_drafting_token_id=args.draft_token_id,
        num_query_per_req=num_query_per_req,
        num_speculative_steps=args.spec_steps,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=args.max_model_len,
        sample_from_anchor=args.sample_from_anchor,
    )
    signature = inspect.signature(dflash_speculator.prepare_dflash_inputs)
    if "temperature" in signature.parameters:
        kwargs.update(
            temperature=tensors["temperature"],
            seeds=tensors["seeds"],
            input_temperature=tensors["input_temperature"],
            input_seeds=tensors["input_seeds"],
        )

    def run() -> None:
        dflash_speculator.prepare_dflash_inputs(**kwargs)

    return run


def main() -> None:
    args = parse_args()
    scheduled_tokens = resolve_scheduled_tokens(args)
    num_query_per_req, max_num_reqs, max_num_tokens = resolve_sizes(
        args, scheduled_tokens
    )
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This workload requires an available Ascend NPU")

    dflash_speculator._prepare_dflash_inputs_kernel = (
        _prepare_dflash_inputs_kernel_ascend
    )
    input_buffers, input_batch, tensors = make_inputs(
        args, scheduled_tokens, num_query_per_req, max_num_reqs, max_num_tokens
    )
    run = make_call(
        args,
        input_buffers,
        input_batch,
        tensors,
        num_query_per_req,
        max_num_reqs,
        max_num_tokens,
    )

    # Compile before the repeated workload so msprof mainly sees kernel execution.
    run()
    torch.npu.synchronize()

    for _ in range(args.repeat):
        run()
    torch.npu.synchronize()

    layout = "DSpark" if args.sample_from_anchor else "DFlash"
    print(
        f"prepare_dflash_inputs completed: layout={layout}, "
        f"num_reqs={args.num_reqs}, context_len={args.context_len}, "
        f"spec_steps={args.spec_steps}, repeat={args.repeat}"
    )


if __name__ == "__main__":
    main()
