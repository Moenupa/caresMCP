"""Reproduce recorded benign, malignant, and normal scores on real mammograms."""

import math

import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.mammography.umair import classify_mammogram_patch

pytestmark = pytest.mark.slow

EXPECTED_LABELS = ("benign", "malignant", "normal")

# Recorded outputs from a deterministic CPU run. The checkpoint reports 99.6%
# test accuracy on its CLAHE-preprocessed patches; this full screening view
# keeps the recorded benign top class.
EXPECTED_SCORES = {
    "mlo_scattered_density": {"benign": 0.9972, "malignant": 0.0005, "normal": 0.0023},
}
EXPECTED_TOP = {
    "mlo_scattered_density": "benign",
}


@pytest.mark.parametrize("image", tuple(EXPECTED_SCORES))
@skip_on_exception(ImportError, OSError)
@save_returns()
def test_classify_mammogram_patch(image: str, request: pytest.FixtureRequest):
    """Score real mammograms with recorded calibrated class probabilities."""
    scores = classify_mammogram_patch(request.getfixturevalue(image))

    # Cheap contract checks first.
    assert tuple(scores) == EXPECTED_LABELS
    assert all(
        isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
        for score in scores.values()
    )
    assert math.isclose(sum(scores.values()), 1.0, abs_tol=1e-3)
    # Then reproduce the recorded scores and ranking.
    assert scores == pytest.approx(EXPECTED_SCORES[image], rel=1e-2)
    assert max(scores, key=scores.__getitem__) == EXPECTED_TOP[image]
    return scores
