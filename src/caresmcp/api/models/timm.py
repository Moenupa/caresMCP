from functools import lru_cache
from typing import TYPE_CHECKING

from caresmcp.constants import DEVICE, LRU_MAX_SIZE

if TYPE_CHECKING:
    from torch.nn import Module


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_timm_model(
    model_name: str,
    pretrained: bool = True,
    **kwargs,
) -> "Module":
    """Load and cache a timm model for inference."""
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    model = model.to(DEVICE)
    model.eval()
    return model
