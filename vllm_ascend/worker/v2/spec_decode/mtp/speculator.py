# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm_ascend.worker.v2.spec_decode.eagle.speculator import (
    AscendEagleSpeculator,
)


class AscendMTPSpeculator(AscendEagleSpeculator):
    """Ascend eager speculator for standard MTP draft models.

    Standard MTP models return one hidden-state tensor, while EAGLE models
    return ``(last_hidden_states, hidden_states)``. The remaining Ascend draft
    loop and attention-metadata handling are shared with EAGLE in this branch.
    """

    @property
    def model_returns_tuple(self) -> bool:
        return False
