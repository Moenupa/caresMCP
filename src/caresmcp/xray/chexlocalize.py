"""Evaluate and segment chest X-ray saliency maps with CheXlocalize metrics."""

from io import BytesIO
from typing import Annotated

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.imageio import convert_grayscale
from caresmcp.registry import TASK_EVAL, TASK_SEG, XRAY

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


def _normalized_saliency(image: bytes) -> np.ndarray:
    return convert_grayscale(image).astype(np.float64) / 255.0


@XRAY.register("chexlocalize_evaluate")
@TASK_EVAL.register("chexlocalize_evaluate")
def chexlocalize_evaluate(
    saliency_map: Annotated[bytes, SALIENCY_MAP_FIELD],
    ground_truth_mask: Annotated[bytes, GROUND_TRUTH_MASK_FIELD],
    saliency_threshold: Annotated[float, SALIENCY_THRESHOLD_FIELD] = 0.5,
) -> dict[str, float]:
    """Measure saliency overlap and peak localization against a reference mask.

    The saliency map and reference mask must have the same dimensions. Every
    nonzero reference pixel is positive; saliency is thresholded on a normalized
    [0, 1] scale.

    Returns:
        ``intersection_over_union``: overlap divided by union of the thresholded
        saliency and reference masks, in [0, 1]. ``pointing_game_hit``: 1 if the
        saliency maximum lies inside the reference mask, otherwise 0. Ties use
        the first maximum in row-major order, including constant maps.

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """

    saliency = _normalized_saliency(saliency_map)
    reference = convert_grayscale(ground_truth_mask)
    if saliency.shape != reference.shape:
        raise ValueError("saliency_map and ground_truth_mask must have the same size")
    reference_mask = reference > 0
    reference_pixels = int(reference_mask.sum())
    if reference_pixels == 0:
        raise ValueError("ground_truth_mask must contain at least one positive pixel")

    predicted_mask = saliency >= saliency_threshold
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
    """Convert a chest X-ray saliency map into a binary localization mask.

    ``threshold`` is a cutoff on the normalized [0, 1] saliency scale.
    ``smoothing_kernel_size`` sets the square neighborhood width in pixels;
    values of 0 or 1 leave the map unsmoothed.

    Returns:
        A PNG mask with 255 for saliency at or above the cutoff and 0 elsewhere.

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """

    values = _normalized_saliency(saliency_map)

    if smoothing_kernel_size > 1:
        values = cv2.blur(
            values,
            (smoothing_kernel_size, smoothing_kernel_size),
        )
    mask = np.uint8(values >= threshold) * 255
    output = BytesIO()
    Image.fromarray(mask, mode="L").save(output, format="PNG")
    return output.getvalue()
