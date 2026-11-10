"""Cached FleXray flagship loading."""

from typing import Any

from caresmcp.constants import DEVICE


def get_flexray_model() -> Any:
    """Load the released flagship anatomical segmentation bundle."""
    from fxr.inference import FleXraySegmenter

    return FleXraySegmenter.from_pretrained(
        "VictorButoi/flexray", subfolder="members/flux0375", device=DEVICE
    )
