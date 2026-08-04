# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendMambaHybridModelState(MambaHybridModelState):
    """Mamba-hybrid state using Ascend attention metadata builders."""

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        max_seq_len = (
            self.max_model_len
            if for_capture
            else seq_lens_cpu_upper_bound[:num_reqs].max().item()
        )

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            if input_batch.num_draft_tokens_per_req is not None:
                has_draft_tokens = input_batch.num_draft_tokens_per_req > 0
                spec_decode_mask = has_draft_tokens & ~input_batch.is_prefilling_np
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    input_batch.num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(
                num_decode_draft_tokens_np
            )

        model_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=model_metadata,
            for_cudagraph_capture=for_capture,
        )
        return self.attn_metadata
