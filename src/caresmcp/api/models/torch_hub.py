from functools import lru_cache
from typing import TYPE_CHECKING

from caresmcp.constants import DEVICE, LRU_MAX_SIZE

if TYPE_CHECKING:
    from torch.nn import Module


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_torch_checkpoint(
    repo_or_dir: str,
    model: str,
    **kwargs,
) -> "Module":
    """Load and cache a PyTorch Hub model for inference."""
    import torch

    model = torch.hub.load(
        repo_or_dir,
        model,
        trust_repo=True,
        skip_validation=True,
        **kwargs,
    )
    model = model.to(DEVICE)
    model.eval()
    return model
