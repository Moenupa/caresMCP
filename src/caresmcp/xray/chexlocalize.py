"""Saliency map utilities for chest X-ray localization evaluation."""

from io import BytesIO
from typing import Any

from caresmcp.registry import TASK_SEG, XRAY


@XRAY.register("chexlocalize_evaluate_saliency_localization")
def chexlocalize_evaluate_saliency_localization(
    saliency_map: bytes,
    ground_truth_mask: bytes,
    saliency_threshold: float = 0.5,
) -> dict[str, float]:
    """Score one saliency map against a radiologist reference mask.

    This computes the per-image intersection-over-union and pointing-game hit
    used in saliency localization evaluation. The threshold converts the
    grayscale saliency map into a binary segmentation; the pointing-game
    location is the saliency map's maximum-valued pixel.

    Args:
        saliency_map: Grayscale PNG or JPEG bytes containing saliency values.
        ground_truth_mask: Grayscale reference mask bytes; nonzero pixels are
            treated as the localized finding.
        saliency_threshold: Saliency cutoff in the range [0, 1].

    Returns:
        Per-image intersection-over-union and pointing-game hit, each in [0, 1].

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

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


@XRAY.register("chexlocalize_segment_saliency_map")
@TASK_SEG.register("chexlocalize_segment_saliency_map")
def chexlocalize_segment_saliency_map(
    saliency_map: bytes,
    threshold: float = 0.5,
    smoothing_kernel_size: int = 0,
) -> bytes:
    """Convert a grayscale chest X-ray saliency map to a binary mask.

    Args:
        saliency_map: Grayscale PNG or JPEG bytes containing saliency values.
        threshold: Saliency cutoff in the range [0, 1].
        smoothing_kernel_size: Optional square box-filter width in pixels;
            zero disables smoothing.

    Returns:
        A binary PNG mask with foreground value 255 and background value 0.

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError

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
    Image.fromarray(mask, mode="L").save(output, format="PNG")  # ty: ignore[invalid-argument-type]
    return output.getvalue()
