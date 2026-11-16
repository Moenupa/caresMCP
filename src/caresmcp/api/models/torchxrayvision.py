"""Cached TorchXRayVision model loading."""

from functools import lru_cache
from typing import Any

from caresmcp.constants import LRU_MAX_SIZE, MODEL_CACHE_DIR

from ..io import device_copy


def get_torchxrayvision_model(weights: str, *, apply_sigmoid: bool = False) -> Any:
    """Load a TorchXRayVision model for inference."""
    return device_copy(_get_torchxrayvision_model(weights, apply_sigmoid=apply_sigmoid))


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_torchxrayvision_model(weights: str, *, apply_sigmoid: bool = False) -> Any:
    """Load and cache a TorchXRayVision model on CPU."""
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
    return model
