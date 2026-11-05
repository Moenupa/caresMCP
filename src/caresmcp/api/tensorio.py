"""Tensor validation and conversion to labeled classification scores."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torch import Tensor


def get_classification_scores(
    logits: "Tensor",
    labels: Mapping[Any, str],
    score_mode: Literal["softmax", "sigmoid"],
) -> dict[str, float]:
    """Convert finite single-item logits to uniquely labeled scores."""
    import torch

    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] == 0:
        raise ValueError("classifier must return [1, classes] logits")
    if not torch.isfinite(logits).all():
        raise ValueError("classifier returned non-finite logits")
    names: list[str] = [
        labels.get(i) or labels.get(str(i)) or f"unnamed_{i}"
        for i in range(logits.shape[1])
    ]
    values = logits.float()
    if score_mode == "softmax":
        values = values.softmax(dim=-1)
    elif score_mode == "sigmoid":
        values = values.sigmoid()
    return dict(zip(names, values[0].detach().cpu().tolist(), strict=True))
