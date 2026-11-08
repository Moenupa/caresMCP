"""Cached TorchXRayVision model loading."""

from functools import lru_cache
from typing import Any

from caresmcp.constants import DEVICE, LRU_MAX_SIZE, MODEL_CACHE_DIR


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_torchxrayvision_model(weights: str, *, apply_sigmoid: bool = False) -> Any:
    """Load and cache a TorchXRayVision model for inference."""
    import torchxrayvision as xrv

    if weights == "anatomy":
        model = xrv.baseline_models.chestx_det.PSPNet(
            cache_dir=MODEL_CACHE_DIR.as_posix()
        )
    else:
        factory = (
            xrv.models.ResNet if weights.startswith("resnet") else xrv.models.DenseNet
        )
        model = factory(
            weights=weights,
            apply_sigmoid=apply_sigmoid,
            cache_dir=MODEL_CACHE_DIR.as_posix(),
        )
    return model.to(DEVICE).eval()
