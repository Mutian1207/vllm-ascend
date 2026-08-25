import torch
import torch_npu  # noqa: F401


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    if k is None and p is None:
        return logits
    # use cann ops
    return torch_npu.npu_top_k_top_p(logits, p, k)
