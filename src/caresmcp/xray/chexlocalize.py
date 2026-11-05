"""Saliency map utilities for chest X-ray localization evaluation."""

from io import BytesIO
from typing import Annotated, Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import Field

from caresmcp.registry import TASK_SEG, XRAY

SALIENCY_MAP_FIELD = Field(description="Grayscale saliency map image")
GROUND_TRUTH_MASK_FIELD = Field(
    description="Grayscale reference mask; nonzero pixels mark the finding"
)
SALIENCY_THRESHOLD_FIELD = Field(
    ge=0,
    le=1,
    description="Saliency cutoff in [0, 1] for including foreground pixels",
)
SMOOTHING_KERNEL_SIZE_FIELD = Field(
    ge=0,
    description="Square box-filter width in pixels; zero disables smoothing",
)


@XRAY.register("chexlocalize_evaluate")
def chexlocalize_evaluate(
    saliency_map: Annotated[bytes, SALIENCY_MAP_FIELD],
    ground_truth_mask: Annotated[bytes, GROUND_TRUTH_MASK_FIELD],
    saliency_threshold: Annotated[float, SALIENCY_THRESHOLD_FIELD] = 0.5,
) -> dict[str, float]:
    """Evaluate a saliency map against a radiologist reference mask.

    Inputs must be same-sized grayscale images; nonzero reference pixels mark
    the finding. ``saliency_threshold`` is a cutoff on the [0, 1] saliency scale.

    Returns:
        ``intersection_over_union``: overlap divided by union of the thresholded
        saliency and reference masks, in [0, 1]. ``pointing_game_hit``: 1 if the
        saliency maximum lies inside the reference mask, otherwise 0.

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """

    if not 0 <= saliency_threshold <= 1:
        raise ValueError("saliency_threshold must be between 0 and 1")

    decoded: list[Any] = []
    for label, image_bytes in (
        ("saliency_map", saliency_map),
        ("ground_truth_mask", ground_truth_mask),
    ):
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                rgba = source.convert("RGBA")
                grayscale = np.asarray(rgba.convert("L"), dtype=np.uint8)
                alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
                decoded.append((grayscale * alpha // 255).astype(np.uint8))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"{label} must contain a readable image") from exc

    saliency, reference = decoded
    if saliency.shape != reference.shape:
        raise ValueError("saliency_map and ground_truth_mask must have the same size")
    reference_mask = reference > 0
    reference_pixels = int(reference_mask.sum())
    if reference_pixels == 0:
        raise ValueError("ground_truth_mask must contain at least one positive pixel")

    predicted_mask = saliency >= round(saliency_threshold * 255)
    intersection = int(np.logical_and(predicted_mask, reference_mask).sum())
    union = int(np.logical_or(predicted_mask, reference_mask).sum())
    peak_y, peak_x = np.unravel_index(int(saliency.argmax()), saliency.shape)
    return {
        "intersection_over_union": float(intersection / union) if union else 0.0,
        "pointing_game_hit": float(reference_mask[peak_y, peak_x]),
    }


@XRAY.register("chexlocalize_segment")
@TASK_SEG.register("chexlocalize_segment")
def chexlocalize_segment(
    saliency_map: Annotated[bytes, SALIENCY_MAP_FIELD],
    threshold: Annotated[float, SALIENCY_THRESHOLD_FIELD] = 0.5,
    smoothing_kernel_size: Annotated[int, SMOOTHING_KERNEL_SIZE_FIELD] = 0,
) -> bytes:
    """Return a binary mask for a grayscale chest X-ray saliency map.

    ``threshold`` is a cutoff on the [0, 1] saliency scale.
    ``smoothing_kernel_size`` is the neighborhood width in pixels; 0 disables
    smoothing.

    Returns:
        A PNG mask with 255 for saliency at or above the cutoff and 0 elsewhere.

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if smoothing_kernel_size < 0:
        raise ValueError("smoothing_kernel_size must not be negative")

    try:
        with Image.open(BytesIO(saliency_map)) as source:
            rgba = source.convert("RGBA")
            grayscale = np.asarray(rgba.convert("L"), dtype=np.uint8)
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
            values = (grayscale * alpha // 255).astype(np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("saliency_map must contain a readable image") from exc

    if smoothing_kernel_size > 1:
        values = cv2.blur(
            values,
            (smoothing_kernel_size, smoothing_kernel_size),
        )
    mask = np.uint8(values >= threshold) * 255
    output = BytesIO()
    Image.fromarray(mask, mode="L").save(output, format="PNG")
    return output.getvalue()
