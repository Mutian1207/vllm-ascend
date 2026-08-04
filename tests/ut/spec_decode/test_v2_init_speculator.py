from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def test_init_speculator_uses_mtp_before_eagle() -> None:
    """MTP is part of use_eagle(), but its model has a different return type."""
    speculative_config = SimpleNamespace(
        method="mtp",
        use_dflash=lambda: False,
        use_eagle=lambda: True,
        use_gemma4_mtp=lambda: False,
        use_step3p5_mtp=lambda: False,
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    expected = MagicMock()

    with patch(
        "vllm_ascend.worker.v2.spec_decode.mtp.speculator.AscendMTPSpeculator",
        return_value=expected,
    ) as mtp_cls:
        result = init_speculator(vllm_config, torch.device("cpu"))

    assert result is expected
    mtp_cls.assert_called_once_with(vllm_config, torch.device("cpu"))
