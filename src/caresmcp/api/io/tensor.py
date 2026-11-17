"""Pytorch support for device and tensors."""

from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

from caresmcp.constants import DEVICE

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


def torch_autocast(dtype=None):
    if DEVICE == "cpu":
        return nullcontext()
    else:
        from torch import autocast

        return autocast(device_type=DEVICE, dtype=dtype)


def device_copy[ModuleT: "Module"](model: ModuleT) -> ModuleT:
    """Return an independent inference-ready copy of a cached model.

    The cache keeps models on the host; each call receives a deep copy in
    evaluation mode, so callers can attach hooks or mutate the module without
    affecting concurrent calls.
    """
    return deepcopy(model).to(DEVICE).eval()


def device_to(tensor: "Tensor") -> "Tensor":
    return tensor.to(DEVICE)


def get_classification_scores(
    logits: "Tensor",
    labels: Mapping[Any, str],
    score_mode: Literal["softmax", "sigmoid", "identity"],
    *,
    skip_empty_labels: bool = False,
) -> dict[str, float]:
    """Convert single-item outputs to labeled scores rounded to four decimals.

    Use ``identity`` for scores already normalized by the model. Set
    ``skip_empty_labels`` when empty labels identify unsupported output channels;
    otherwise missing labels receive an index-based name.
    """
    import torch

    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] == 0:
        raise ValueError("classifier must return [1, classes] logits")
    if not torch.isfinite(logits).all():
        raise ValueError("classifier returned non-finite logits")
    names: list[str] = [
        labels.get(i) or labels.get(str(i)) or f"unnamed_label_{i}"
        for i in range(logits.shape[1])
    ]
    values = logits.float()
    if score_mode == "softmax":
        values = values.softmax(dim=-1)
    elif score_mode == "sigmoid":
        values = values.sigmoid()
    return {
        name: round(score, 4)
        for i, (name, score) in enumerate(
            zip(names, values[0].detach().cpu().tolist(), strict=True)
        )
        if not skip_empty_labels or labels.get(i) or labels.get(str(i))
    }
