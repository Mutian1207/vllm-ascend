# prepare_dflash_inputs

## Description

- **Function**: Prepares the metadata and fixed-capacity buffers required by DFlash speculative decoding on Ascend NPU. For each request, the operator converts the current target-model token span into draft-model context KV positions/slots, constructs the next DFlash query tokens and query slots, builds sampling mappings, copies per-request sampling state, and pads graph-visible buffers. The file provides two Ascend implementations: `prepare_dflash_inputs_ascend` for the non-DCP DFlash ABI and `prepare_dflash_inputs_ascend_dcp` for the DCP-aware ABI.
- **Formula**:
    - Request context range:
      `ctx_start = query_start_loc[req]`,
      `ctx_end = query_start_loc[req + 1]`,
      `valid_ctx_end = ctx_end - num_rejected[req]`,
      `last_valid_pos = positions[valid_ctx_end - 1]`.
    - Non-DCP context/query KV slot:
      `logical_block = position // block_size`,
      `physical_block = block_table[req, logical_block]`,
      `slot = physical_block * block_size + position % block_size`.
    - DCP context/query KV slot:
      `logical_block = position // (block_size * cp_size)`;
      the physical block is read from `block_table`, then the rank-local slot is computed from `cp_rank`, `cp_size`, and `cp_interleave`. Positions not owned by the current rank and null physical block `0` map to `PAD_SLOT_ID`.
    - Query construction:
      `query_pos = last_valid_pos + 1 + query_offset`;
      query offset `0` uses the request bonus token, and subsequent offsets use `parallel_drafting_token_id`.
    - Sampling mapping:
      `sample_off = 0` when `sample_from_anchor=True`, otherwise `1`;
      each speculative sample maps to the corresponding query row and request-state index.
    - Work distribution:
      each request's context/query/sample range is divided across `workers_per_req` using quotient/remainder partitioning; graph-padding ranges are divided across all launched programs using the same rule.
- **Algorithm flow** (processed row by row, independently):
  1. Read `input_batch.num_scheduled_tokens` and the Ascend VectorCore count. Compute a request-local worker count that is at least the upstream coverage requirement and targets the available VectorCore parallelism.
  2. Launch a 2-D grid `(num_reqs, workers_per_req)`.
  3. For each request, split the context range across its workers with quotient/remainder partitioning. Each worker processes one continuous context sub-range and writes `context_positions` and `context_slot_mapping`.
  4. Independently split `num_query_per_req` across the same request workers. Build the bonus/mask query token IDs, absolute query positions, and query KV slots.
  5. Independently split `num_speculative_steps` across request workers and write `sample_indices`, `sample_pos`, and `sample_idx_mapping`.
  6. Worker `0` of each request writes the per-request scalar outputs `query_start_loc`, `seq_lens`, `temperature`, and `seeds`.
  7. Flatten the full launch grid and evenly distribute the graph-padding ranges for `query_start_loc`, `seq_lens`, sample buffers, and `query_slot_mapping`.
  8. The DCP-aware implementation additionally initializes rejected context suffix rows to `position=0/PAD_SLOT_ID`, rejects null physical block `0`, computes rank-local KV slots, and clamps `seq_lens` to `max_model_len`.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950

## Parameters

> [!NOTE]
>
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input_buffers` | Output | Preallocated DFlash query buffers. `input_ids:[max_num_tokens]`, `positions:[max_num_tokens]`, `query_start_loc:[max_num_reqs+1]`, `seq_lens:[max_num_reqs]` are written in place. | `input_ids`: int32; `positions`: int64; `query_start_loc`/`seq_lens`: int32 | ND |
| `query_slot_mapping` | Output | Physical KV slot for each draft query; valid prefix length is `num_reqs * num_query_per_req`, remaining graph-visible entries are padded with `PAD_SLOT_ID`. | int32 | ND |
| `context_positions` | Output | Absolute target positions used to precompute/store draft context KV. Valid span length equals the flattened target context span. In the DCP-aware ABI, rejected suffix rows are written as `0`. | int64 | ND |
| `context_slot_mapping` | Output | Physical draft KV slot for each target context token. In the DCP-aware ABI, rejected/non-resident/non-local rows are `PAD_SLOT_ID`. | int32 | ND |
| `sample_indices` | Output | Query-row indices selected from draft hidden states for speculative sampling. Shape `[max_num_reqs * num_speculative_steps]`. | int64 | ND |
| `sample_pos` | Output | Absolute sampled-token positions corresponding to `sample_indices`. Shape `[max_num_reqs * num_speculative_steps]`. | int64 | ND |
| `sample_idx_mapping` | Output | Maps each speculative sample to its persistent request-state index. Padded entries are `-1`. | int32 | ND |
| `temperature` | Output | Per-request sampling temperature copied to the persistent request-state slot. Shape `[max_num_reqs]`. | float32 | ND |
| `seeds` | Output | Per-request sampling seed copied to the persistent request-state slot. Shape `[max_num_reqs]`. | int64 | ND |
| `input_batch` | Input | Current target batch. The operator reads `num_reqs`, `num_scheduled_tokens`, `positions`, `query_start_loc`, and `idx_mapping`. | object; member tensors: int64/int32; `num_scheduled_tokens`: ndarray | N/A |
| `num_sampled` | Input | Number/state flag of sampled tokens for each active request; selects `last_sampled` vs `next_prefill_tokens` as the bonus token source. | int32 | ND |
| `num_rejected` | Input | Number of rejected speculative tokens at the end of each request's current target span. | int32 | ND |
| `last_sampled` | Input | Persistent last-sampled token per request-state slot. | int64 | ND |
| `next_prefill_tokens` | Input | Persistent next-prefill token per request-state slot, used when no sampled token is available. | int32 | ND |
| `input_temperature` | Input | Source per-request sampling temperature indexed by `input_batch.idx_mapping`. | float32 | ND |
| `input_seeds` | Input | Source per-request sampling seed indexed by `input_batch.idx_mapping`. | int64 | ND |
| `block_table` | Input | Request-to-physical-KV-block table, shape `[max_num_reqs, max_num_blocks]`. | int32 | ND |
| `block_size` | Input/Attribute | Number of tokens in one local KV block. Used in logical-block and physical-slot calculation. | int | scalar |
| `cp_rank` | Input/Attribute | DCP rank used by `prepare_dflash_inputs_ascend_dcp`; not present in the non-DCP ABI. | int | scalar |
| `cp_size` | Input/Attribute | DCP world size used by `prepare_dflash_inputs_ascend_dcp`; not present in the non-DCP ABI. | int | scalar |
| `cp_interleave` | Input/Attribute | Number of consecutive tokens assigned to a DCP rank before round-robin rotation; not present in the non-DCP ABI. | int | scalar |
| `parallel_drafting_token_id` | Input/Attribute | Token ID placed at non-anchor DFlash query positions. | int | scalar |
| `num_query_per_req` | Input/Attribute | Number of draft query rows produced per request. For regular DFlash this is typically `1 + num_speculative_steps`. | int | scalar |
| `num_speculative_steps` | Input/Attribute | Number of speculative samples produced per request. | int | scalar |
| `max_num_reqs` | Input/Attribute | Fixed request capacity of graph-visible output buffers. | int | scalar |
| `max_num_tokens` | Input/Attribute | Fixed token capacity of graph-visible query/context buffers. | int | scalar |
| `max_model_len` | Input/Attribute | Maximum model sequence length. Query positions are clamped to `max_model_len - 1`; DCP-aware `seq_lens` is clamped to `max_model_len`. | int | scalar |
| `sample_from_anchor` | Input/Attribute | If `False`, the bonus token is an anchor and sampling starts from query offset `1`; if `True`, every query position is sampled and predicts the next position. | bool | scalar |

## Constraints

- The operator is inference-only and requires Ascend NPU Triton execution.
- `input_batch.num_reqs` must be greater than `0` and not exceed `max_num_reqs`.
- `input_batch.query_start_loc` must contain at least `num_reqs + 1` int32 entries, be non-decreasing, and delimit the flattened `input_batch.positions` tensor.
- `input_batch.positions` is int64. Every valid absolute position used for block-table lookup must map to a logical block index smaller than `block_table.shape[1]`; the implementation clamps the index to the final block-table column for safety, matching the upstream contract.
- `input_batch.idx_mapping` is int32. The first `num_reqs` entries must index valid request-state slots in `[0, max_num_reqs)`.
- `num_sampled` and `num_rejected` are int32 and must contain at least `num_reqs` valid entries. `0 <= num_rejected[req] < ctx_len[req]` so that `last_valid_pos` exists.
- `last_sampled` and `input_seeds` use int64; `next_prefill_tokens` uses int32; `input_temperature` uses float32. Their flattened storage must contain at least `max_num_reqs` request-state entries.
- `block_table` is int32 with shape `[max_num_reqs, max_num_blocks]`; `block_size > 0`.
- `query_slot_mapping` and `context_slot_mapping` are int32. `context_positions` and query positions are int64. `sample_indices`/`sample_pos` are int64 and `sample_idx_mapping` is int32.
- `num_query_per_req > 0`, `num_speculative_steps > 0`, `max_num_tokens >= num_reqs * num_query_per_req`, and `max_model_len > 0`.
- The wrapper computes `BLOCK_SIZE = min(256, next_power_of_2(max_target_query_len + num_query_per_req))` and chooses enough request-local workers to preserve full context coverage while targeting the detected VectorCore parallelism.
- The non-DCP implementation preserves the legacy DFlash slot-mapping semantics.
- For the DCP-aware implementation, `cp_size >= 1`, `0 <= cp_rank < cp_size`, and `cp_interleave >= 1`. Physical block `0` is the null block and is never returned as a writable KV slot. Rejected context suffix rows and positions not owned by the current DCP rank are written with `PAD_SLOT_ID`.
- The operator writes padding for graph-visible fixed-capacity buffers and is compatible with eager execution and graph replay/capture paths that consume those buffers.

## Origin and Differences

- **Origin**: Optimized from `vllm.v1.worker.gpu.spec_decode.dflash.speculator.prepare_dflash_inputs` and `_prepare_dflash_inputs_kernel`. The non-DCP implementation follows the legacy upstream DFlash ABI; the DCP-aware implementation follows the current upstream ABI and DCP slot-mapping semantics.
- **Differences**:
    - NPU adaptation for performance: replaces the previous Ascend one-effective-program-per-request scalar execution with a VectorCore-aware 2-D launch. Context, query, and sample domains are independently partitioned with quotient/remainder balancing, while graph padding is balanced across the entire launch grid. This removes long serial loops and the single-last-request padding bottleneck without hard-coding a device core count;
    - Modified for a specific vllm-ascend logic or different input parameters: two explicit wrappers/kernels are provided because upstream DFlash has non-DCP and DCP-aware ABIs. The DCP-aware Ascend implementation preserves current upstream rejected-suffix initialization, null-block protection, DCP rank-local KV-slot calculation, and `seq_lens` clamping.

## Test Cases

The accuracy test covers both captured DFlash inference shapes and the DCP-specific semantic regression path. The captured business shapes are:

- `num_reqs=8`, `target_positions.shape=[1626]`, `num_query_per_req=9`, `num_speculative_steps=8`, `max_num_reqs=64`, `max_num_tokens=8192`, `block_size=128`;
- `num_reqs=64`, `target_positions.shape=[576]`, `num_query_per_req=9`, `num_speculative_steps=8`, `max_num_reqs=64`, `max_num_tokens=8192`, `block_size=128`.

The test uses deterministic legal metadata with these captured shapes and compares all operator-produced valid/padded regions against an independent Python/Torch reference. Because the operator performs integer index/address transformation and direct sampling-state copies, outputs are required to be bit-exact (`rtol=0, atol=0`). The same UT file covers `prepare_dflash_inputs_ascend`, `prepare_dflash_inputs_ascend_dcp` with `cp_size=1`, and a targeted DCP `cp_size=2` rejected-context/local-slot regression case.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_dflash_inputs.py
```

## Example

A worked example of this template is committed alongside it in the same branch:

- **Operator doc**: `vllm_ascend/ops/triton/docs/fused_qkvzba_split_reshape.md`
- **Accuracy test**: `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_qkvzba_split_reshape_cat.py`
