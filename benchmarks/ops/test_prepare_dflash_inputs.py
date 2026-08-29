# SPDX-License-Identifier: Apache-2.0
"""Run and validate Ascend's patched prepare_dflash_inputs kernel."""

import argparse
import inspect
from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
# Match the normal plugin initialization order. Importing the worker module first
# makes device_op and ops.__init__ enter each other while both are half initialized.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel_ascend,
)


SENTINEL_INT = -777
SENTINEL_FLOAT = -777.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Ascend prepare_dflash_inputs kernel once and compare all "
            "outputs with a CPU reference."
        )
    )
    parser.add_argument("--num-reqs", type=int, default=2)
    parser.add_argument(
        "--context-len",
        type=int,
        default=8,
        help="Number of target tokens scheduled for each request.",
    )
    parser.add_argument("--spec-steps", type=int, default=3)
    parser.add_argument(
        "--sample-from-anchor",
        action="store_true",
        help="Use the default DSpark layout. Without it, use DFlash layout.",
    )
    parser.add_argument(
        "--num-rejected",
        type=int,
        default=0,
        help="Rejected target-tail tokens per request.",
    )
    parser.add_argument(
        "--chunked-prefill",
        action="store_true",
        help="Set num_sampled=0 so the anchor comes from next_prefill_tokens.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-num-reqs",
        type=int,
        default=None,
        help="Capacity of persistent request buffers; defaults to num_reqs + 2.",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=None,
        help="Capacity of token buffers; defaults to the minimum required size.",
    )
    parser.add_argument("--draft-token-id", type=int, default=999)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print complete output tensors after validation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.num_reqs <= 0:
        raise ValueError("--num-reqs must be positive")
    if args.context_len <= 0:
        raise ValueError("--context-len must be positive")
    if args.spec_steps <= 0:
        raise ValueError("--spec-steps must be positive")
    if not 0 <= args.num_rejected < args.context_len:
        raise ValueError("--num-rejected must be in [0, context_len)")
    if args.block_size <= 0 or args.max_model_len <= 0:
        raise ValueError("--block-size and --max-model-len must be positive")

    num_query_per_req = (
        args.spec_steps if args.sample_from_anchor else args.spec_steps + 1
    )
    max_num_reqs = args.max_num_reqs or args.num_reqs + 2
    if max_num_reqs < args.num_reqs:
        raise ValueError("--max-num-reqs cannot be smaller than --num-reqs")

    required_tokens = max(
        args.num_reqs * args.context_len,
        args.num_reqs * num_query_per_req,
    )
    max_num_tokens = args.max_num_tokens or required_tokens + num_query_per_req
    if max_num_tokens < required_tokens:
        raise ValueError(
            f"--max-num-tokens must be at least {required_tokens} for this case"
        )
    return num_query_per_req, max_num_reqs, max_num_tokens


def make_tensors(
    args: argparse.Namespace,
    num_query_per_req: int,
    max_num_reqs: int,
    max_num_tokens: int,
) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, torch.Tensor]]:
    device = torch.device(args.device)
    total_context_tokens = args.num_reqs * args.context_len

    # Each request has its own logical positions [0, ..., context_len - 1].
    target_positions_cpu = torch.arange(args.context_len, dtype=torch.int64).repeat(
        args.num_reqs
    )
    query_start_loc_cpu = torch.arange(
        0,
        total_context_tokens + 1,
        args.context_len,
        dtype=torch.int32,
    )

    # Reverse the active persistent-state seats so idx_mapping is genuinely used.
    idx_mapping_cpu = torch.arange(
        args.num_reqs - 1, -1, -1, dtype=torch.int32
    )
    input_batch = SimpleNamespace(
        num_reqs=args.num_reqs,
        num_scheduled_tokens=np.full(
            args.num_reqs, args.context_len, dtype=np.int32
        ),
        positions=target_positions_cpu.to(device),
        query_start_loc=query_start_loc_cpu.to(device),
        idx_mapping=idx_mapping_cpu.to(device),
    )

    def full(size: int | tuple[int, ...], value: int, dtype: torch.dtype):
        return torch.full(
            (size,) if isinstance(size, int) else size,
            value,
            dtype=dtype,
            device=device,
        )

    input_buffers = SimpleNamespace(
        input_ids=full(max_num_tokens, SENTINEL_INT, torch.int32),
        positions=full(max_num_tokens, SENTINEL_INT, torch.int64),
        query_start_loc=full(max_num_reqs + 1, SENTINEL_INT, torch.int32),
        seq_lens=full(max_num_reqs, SENTINEL_INT, torch.int32),
    )

    max_num_blocks = (args.max_model_len + args.block_size - 1) // args.block_size
    # Unique physical block IDs make incorrect row/block lookups visible.
    block_table_cpu = torch.arange(
        args.num_reqs * max_num_blocks, dtype=torch.int32
    ).view(args.num_reqs, max_num_blocks)

    tensors = {
        "query_slot_mapping": full(max_num_tokens, SENTINEL_INT, torch.int64),
        "context_positions": full(max_num_tokens, SENTINEL_INT, torch.int64),
        "context_slot_mapping": full(max_num_tokens, SENTINEL_INT, torch.int64),
        "sample_indices": full(
            max_num_reqs * args.spec_steps, SENTINEL_INT, torch.int64
        ),
        "sample_pos": full(
            max_num_reqs * args.spec_steps, SENTINEL_INT, torch.int64
        ),
        "sample_idx_mapping": full(
            max_num_reqs * args.spec_steps, SENTINEL_INT, torch.int32
        ),
        "temperature": torch.full(
            (max_num_reqs,), SENTINEL_FLOAT, dtype=torch.float32, device=device
        ),
        "seeds": full(max_num_reqs, SENTINEL_INT, torch.int64),
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
        ),
        "next_prefill_tokens": torch.arange(
            2000, 2000 + max_num_reqs, dtype=torch.int32, device=device
        ),
        "input_temperature": torch.arange(
            max_num_reqs, dtype=torch.float32, device=device
        )
        / 10
        + 0.5,
        "input_seeds": torch.arange(
            3000, 3000 + max_num_reqs, dtype=torch.int64, device=device
        ),
        "block_table": block_table_cpu.to(device),
    }
    return input_buffers, input_batch, tensors


def cpu_reference(
    args: argparse.Namespace,
    input_buffers: SimpleNamespace,
    input_batch: SimpleNamespace,
    tensors: dict[str, torch.Tensor],
    num_query_per_req: int,
    max_num_reqs: int,
    max_num_tokens: int,
) -> dict[str, torch.Tensor]:
    source = {
        "target_positions": input_batch.positions.cpu(),
        "target_query_start_loc": input_batch.query_start_loc.cpu(),
        "idx_mapping": input_batch.idx_mapping.cpu(),
        **{name: tensor.cpu() for name, tensor in tensors.items()},
    }
    expected = {
        "input_ids": input_buffers.input_ids.cpu().clone(),
        "positions": input_buffers.positions.cpu().clone(),
        "query_start_loc": input_buffers.query_start_loc.cpu().clone(),
        "seq_lens": input_buffers.seq_lens.cpu().clone(),
        "query_slot_mapping": source["query_slot_mapping"].clone(),
        "context_positions": source["context_positions"].clone(),
        "context_slot_mapping": source["context_slot_mapping"].clone(),
        "sample_indices": source["sample_indices"].clone(),
        "sample_pos": source["sample_pos"].clone(),
        "sample_idx_mapping": source["sample_idx_mapping"].clone(),
        "temperature": source["temperature"].clone(),
        "seeds": source["seeds"].clone(),
    }

    for req_idx in range(args.num_reqs):
        req_state_idx = int(source["idx_mapping"][req_idx])
        ctx_start = int(source["target_query_start_loc"][req_idx])
        ctx_end = int(source["target_query_start_loc"][req_idx + 1])
        valid_ctx_end = ctx_end - int(source["num_rejected"][req_idx])
        last_valid_pos = int(source["target_positions"][valid_ctx_end - 1])
        query_base = req_idx * num_query_per_req

        if int(source["num_sampled"][req_idx]) > 0:
            bonus_token = int(source["last_sampled"][req_state_idx])
        else:
            bonus_token = int(source["next_prefill_tokens"][req_state_idx])

        for ctx_idx in range(ctx_start, ctx_end):
            ctx_pos = int(source["target_positions"][ctx_idx])
            block_num = min(
                ctx_pos // args.block_size, source["block_table"].shape[1] - 1
            )
            block_id = int(source["block_table"][req_idx, block_num])
            expected["context_positions"][ctx_idx] = ctx_pos
            expected["context_slot_mapping"][ctx_idx] = (
                block_id * args.block_size + ctx_pos % args.block_size
            )

        for query_off in range(num_query_per_req):
            query_idx = query_base + query_off
            query_pos = last_valid_pos + 1 + query_off
            block_num = min(
                query_pos // args.block_size,
                source["block_table"].shape[1] - 1,
            )
            block_id = int(source["block_table"][req_idx, block_num])
            expected["input_ids"][query_idx] = (
                bonus_token if query_off == 0 else args.draft_token_id
            )
            expected["positions"][query_idx] = min(
                query_pos, args.max_model_len - 1
            )
            expected["query_slot_mapping"][query_idx] = (
                block_id * args.block_size + query_pos % args.block_size
            )

        sample_off = 0 if args.sample_from_anchor else 1
        for offset in range(sample_off, num_query_per_req):
            sample_idx = req_idx * args.spec_steps + offset - sample_off
            query_idx = query_base + offset
            query_pos = last_valid_pos + 1 + offset
            expected["sample_indices"][sample_idx] = query_idx
            expected["sample_pos"][sample_idx] = (
                query_pos + 1 if args.sample_from_anchor else query_pos
            )
            expected["sample_idx_mapping"][sample_idx] = req_state_idx

        expected["query_start_loc"][req_idx] = query_base
        expected["seq_lens"][req_idx] = (
            last_valid_pos + 1 + num_query_per_req
        )
        if "input_temperature" in source:
            expected["temperature"][req_state_idx] = source["input_temperature"][
                req_state_idx
            ]
            expected["seeds"][req_state_idx] = source["input_seeds"][req_state_idx]

    query_end = args.num_reqs * num_query_per_req
    expected["query_start_loc"][args.num_reqs :] = query_end
    expected["seq_lens"][args.num_reqs :] = 0
    sample_end = args.num_reqs * args.spec_steps
    expected["sample_indices"][sample_end:] = 0
    expected["sample_pos"][sample_end:] = 0
    expected["sample_idx_mapping"][sample_end:] = -1
    expected["query_slot_mapping"][query_end:max_num_tokens] = PAD_SLOT_ID
    return expected


def run_kernel(
    args: argparse.Namespace,
    input_buffers: SimpleNamespace,
    input_batch: SimpleNamespace,
    tensors: dict[str, torch.Tensor],
    num_query_per_req: int,
    max_num_reqs: int,
    max_num_tokens: int,
) -> None:
    # Make replacement explicit: the test must exercise the Ascend kernel even if
    # plugin loading was restricted through VLLM_PLUGINS.
    dflash_speculator._prepare_dflash_inputs_kernel = (
        _prepare_dflash_inputs_kernel_ascend
    )

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
    dflash_speculator.prepare_dflash_inputs(**kwargs)
    torch.npu.synchronize()


def collect_outputs(
    input_buffers: SimpleNamespace, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": input_buffers.input_ids.cpu(),
        "positions": input_buffers.positions.cpu(),
        "query_start_loc": input_buffers.query_start_loc.cpu(),
        "seq_lens": input_buffers.seq_lens.cpu(),
        **{
            name: tensors[name].cpu()
            for name in (
                "query_slot_mapping",
                "context_positions",
                "context_slot_mapping",
                "sample_indices",
                "sample_pos",
                "sample_idx_mapping",
                "temperature",
                "seeds",
            )
        },
    }


def main() -> None:
    args = parse_args()
    num_query_per_req, max_num_reqs, max_num_tokens = validate_args(args)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This test requires an available Ascend NPU")

    input_buffers, input_batch, tensors = make_tensors(
        args, num_query_per_req, max_num_reqs, max_num_tokens
    )
    expected = cpu_reference(
        args,
        input_buffers,
        input_batch,
        tensors,
        num_query_per_req,
        max_num_reqs,
        max_num_tokens,
    )

    run_kernel(
        args,
        input_buffers,
        input_batch,
        tensors,
        num_query_per_req,
        max_num_reqs,
        max_num_tokens,
    )
    actual = collect_outputs(input_buffers, tensors)

    checked = []
    for name, expected_tensor in expected.items():
        # vLLM 0.26 does not copy sampling state in this kernel.
        if name in ("temperature", "seeds") and "temperature" not in inspect.signature(
            dflash_speculator.prepare_dflash_inputs
        ).parameters:
            continue
        torch.testing.assert_close(actual[name], expected_tensor, rtol=0, atol=0)
        checked.append(name)

    layout = "DSpark/anchor-sampling" if args.sample_from_anchor else "DFlash"
    print("prepare_dflash_inputs: PASS")
    print(
        f"layout={layout}, num_reqs={args.num_reqs}, "
        f"context_len={args.context_len}, spec_steps={args.spec_steps}, "
        f"num_query_per_req={num_query_per_req}"
    )
    print(
        f"num_rejected={args.num_rejected}, chunked_prefill={args.chunked_prefill}, "
        f"max_num_reqs={max_num_reqs}, max_num_tokens={max_num_tokens}"
    )
    print(f"validated outputs ({len(checked)}): {', '.join(checked)}")

    if args.show_output:
        for name in checked:
            print(f"{name}={actual[name].tolist()}")


if __name__ == "__main__":
    main()
