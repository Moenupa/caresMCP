"""Load models and checkpoints distributed through PyTorch Hub."""

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from caresmcp.constants import LRU_MAX_SIZE

from ..io import device_copy

if TYPE_CHECKING:
    from torch.nn import Module


@lru_cache(maxsize=LRU_MAX_SIZE)
def load_torch_checkpoint(url: str, *, weights_only: bool = True) -> Mapping[str, Any]:
    """Download and cache a PyTorch Hub checkpoint state dict on CPU."""
    from torch.hub import load_state_dict_from_url

    return load_state_dict_from_url(
        url,
        map_location="cpu",
        weights_only=weights_only,
    )


def get_torch_model(
    repo_or_dir: str,
    model: str,
    **kwargs,
) -> "Module":
    """Load a PyTorch Hub model for inference."""
    return device_copy(_get_torch_model(repo_or_dir, model, **kwargs))


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_torch_model(
    repo_or_dir: str,
    model: str,
    **kwargs,
) -> "Module":
    """Load and cache a PyTorch Hub model on CPU."""
    from torch.hub import load

    net: Module = load(
        repo_or_dir,
        model,
        trust_repo=True,
        skip_validation=True,
        **kwargs,
    )
    return net
