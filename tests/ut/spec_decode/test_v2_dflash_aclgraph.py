import inspect

from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager


def test_dflash_aclgraph_matches_vllm_v024_interface() -> None:
    """Keep the patched graph manager compatible with vLLM v0.24 DFlash."""
    init_params = inspect.signature(DFlashAclGraphManager.__init__).parameters
    assert "causal" in init_params
    assert init_params["causal"].default is False

    ascend_capture = inspect.signature(DFlashAclGraphManager.capture).parameters
    upstream_capture = inspect.signature(DFlashCudaGraphManager.capture).parameters
    assert tuple(ascend_capture) == tuple(upstream_capture)
