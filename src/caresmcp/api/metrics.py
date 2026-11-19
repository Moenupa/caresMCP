"""Overlap metrics for comparing binary masks."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class MaskOverlap:
    """Pixel counts and overlap ratios between two binary masks.

    ``dice`` and ``iou`` are ``None`` when their denominator is zero, so an
    empty prediction against an empty reference has no defined overlap rather
    than a perfect score.
    """

    intersection_pixels: int
    predicted_pixels: int
    reference_pixels: int
    false_positive_pixels: int
    dice: float | None
    iou: float | None


def mask_overlap(predicted: ArrayLike, reference: ArrayLike) -> MaskOverlap:
    """Measure overlap between a predicted mask and a reference mask.

    Args:
        predicted: Candidate foreground mask. Values are coerced to boolean, so
            zero is background and any nonzero value is foreground.
        reference: Reference foreground mask with the same shape as
            ``predicted``.

    Returns:
        Pixel counts together with the Dice coefficient and intersection over
        union.

    Raises:
        ValueError: If the masks have different shapes.
    """
    predicted_mask = np.asarray(predicted, dtype=bool)
    reference_mask = np.asarray(reference, dtype=bool)
    if predicted_mask.shape != reference_mask.shape:
        raise ValueError("Predicted and reference masks must have the same shape")
    intersection = int(np.count_nonzero(predicted_mask & reference_mask))
    predicted_pixels = int(np.count_nonzero(predicted_mask))
    reference_pixels = int(np.count_nonzero(reference_mask))
    union = predicted_pixels + reference_pixels - intersection
    dice_denominator = predicted_pixels + reference_pixels
    return MaskOverlap(
        intersection_pixels=intersection,
        predicted_pixels=predicted_pixels,
        reference_pixels=reference_pixels,
        false_positive_pixels=int(np.count_nonzero(predicted_mask & ~reference_mask)),
        dice=2 * intersection / dice_denominator if dice_denominator else None,
        iou=intersection / union if union else None,
    )
