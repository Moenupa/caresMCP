"""Check the shared mask overlap metric on known pixel arrangements."""

from dataclasses import asdict

import numpy as np
import pytest

from caresmcp.api.metrics import mask_overlap


def test_mask_overlap_counts_partial_overlap():
    """Report intersection, false positives, Dice, and IoU for one overlap."""
    predicted = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    reference = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    overlap = mask_overlap(predicted, reference)
    assert overlap.intersection_pixels == 2
    assert overlap.predicted_pixels == 3
    assert overlap.reference_pixels == 2
    assert overlap.false_positive_pixels == 1
    assert overlap.dice == pytest.approx(2 * 2 / (3 + 2))
    assert overlap.iou == pytest.approx(2 / 3)


def test_mask_overlap_identical_masks_score_one():
    """Treat identical nonempty masks as perfect agreement."""
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    overlap = mask_overlap(mask, mask)
    assert overlap.dice == 1.0
    assert overlap.iou == 1.0
    assert overlap.false_positive_pixels == 0


def test_mask_overlap_disjoint_masks_score_zero():
    """Keep a nonempty prediction with no shared pixels at zero overlap."""
    predicted = np.array([[1, 1], [0, 0]], dtype=bool)
    reference = np.array([[0, 0], [1, 0]], dtype=bool)
    overlap = mask_overlap(predicted, reference)
    assert overlap.intersection_pixels == 0
    assert overlap.false_positive_pixels == 2
    assert overlap.dice == 0.0
    assert overlap.iou == 0.0


def test_mask_overlap_empty_masks_have_no_defined_score():
    """Avoid reporting perfect overlap when both masks are empty."""
    empty = np.zeros((2, 2), dtype=bool)
    overlap = mask_overlap(empty, empty)
    assert overlap.predicted_pixels == 0
    assert overlap.reference_pixels == 0
    assert overlap.false_positive_pixels == 0
    assert overlap.dice is None
    assert overlap.iou is None


def test_mask_overlap_coerces_nonzero_values_to_foreground():
    """Treat any nonzero encoded value as foreground when coercing to boolean."""
    predicted = np.array([[0, 255]], dtype=np.uint8)
    reference = np.array([[False, True]])
    overlap = mask_overlap(predicted, reference)
    assert overlap.predicted_pixels == 1
    assert overlap.intersection_pixels == 1
    assert overlap.dice == 1.0


def test_mask_overlap_serializes_through_asdict():
    """Keep every reported value available to JSON result artifacts."""
    overlap = mask_overlap(np.array([True]), np.array([True]))
    assert asdict(overlap) == {
        "intersection_pixels": 1,
        "predicted_pixels": 1,
        "reference_pixels": 1,
        "false_positive_pixels": 0,
        "dice": 1.0,
        "iou": 1.0,
    }


def test_mask_overlap_rejects_mismatched_shapes():
    """Reject masks whose shapes differ before counting pixels."""
    with pytest.raises(ValueError, match="same shape"):
        mask_overlap(np.zeros((2, 2), dtype=bool), np.zeros((2, 3), dtype=bool))
