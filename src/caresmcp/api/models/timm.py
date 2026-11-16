from functools import lru_cache
from typing import TYPE_CHECKING

from caresmcp.constants import LRU_MAX_SIZE

from ..io import device_copy

if TYPE_CHECKING:
    from torch.nn import Module


def build_timm_model(
    model_name: str,
    pretrained: bool = True,
    **kwargs,
) -> "Module":
    """Build a timm model that the caller owns and may load a checkpoint into."""
    import timm

    return timm.create_model(model_name, pretrained=pretrained, **kwargs)


def get_timm_model(
    model_name: str,
    pretrained: bool = True,
    **kwargs,
) -> "Module":
    """Load a timm model for inference."""
    return device_copy(_get_timm_model(model_name, pretrained, **kwargs))


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_timm_model(
    model_name: str,
    pretrained: bool = True,
    **kwargs,
) -> "Module":
    """Load and cache a timm model on CPU."""
    return build_timm_model(model_name, pretrained, **kwargs)
