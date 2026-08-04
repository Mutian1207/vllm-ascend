import inspect

from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)


def test_dflash_aclgraph_matches_vllm_v024_interface() -> None:
    """Keep the patched graph manager compatible with vLLM v0.24 DFlash."""
    init_params = inspect.signature(DFlashAclGraphManager.__init__).parameters
    assert "causal" in init_params
    assert init_params["causal"].default is False

    ascend_capture = inspect.signature(DFlashAclGraphManager.capture).parameters
    upstream_capture = inspect.signature(DFlashCudaGraphManager.capture).parameters
    assert tuple(ascend_capture) == tuple(upstream_capture)


def test_dflash_speculator_uses_vllm_v024_single_kv_cache_group() -> None:
    set_attn_source = inspect.getsource(AscendDFlashSpeculator.set_attn)
    assert "self.context_slot_mapping" in set_attn_source
    assert "draft_kv_cache_group_ids" not in set_attn_source
