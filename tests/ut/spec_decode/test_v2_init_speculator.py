from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import init_speculator


def test_init_speculator_without_dspark_api() -> None:
    """vLLM v0.24 has EAGLE/DFlash but no SpeculativeConfig.use_dspark."""
    speculative_config = SimpleNamespace(
        method="unsupported",
        use_dflash=lambda: False,
        use_eagle=lambda: False,
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    with pytest.raises(NotImplementedError, match="unsupported is not supported yet"):
        init_speculator(vllm_config, torch.device("cpu"))
