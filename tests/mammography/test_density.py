"""Run the breast density classifier on real mammograms from the model card."""

import math

import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.mammography.density import _LABELS, classify_breast_density


@pytest.mark.slow
@save_returns("mammography_density")
def test_classify_breast_density(mlo_scattered_density: bytes):
    """Score the real right MLO view as scattered fibroglandular density (B)."""
    pytest.importorskip("monai")
    density_scores = classify_breast_density(mlo_scattered_density)

    assert tuple(density_scores) == _LABELS
    assert all(
        isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
        for score in density_scores.values()
    )
    assert density_scores[_LABELS[1]] > 0.5
    assert density_scores[_LABELS[2]] > 0.5
    return density_scores


@pytest.mark.slow
@pytest.mark.parametrize(
    ("image", "expected_label"),
    [
        pytest.param("b_sample", _LABELS[1], id="B"),
        pytest.param("c_sample", _LABELS[2], id="C"),
    ],
)
@save_returns("mammography_density")
def test_classify_breast_density_model_card_samples(
    image: str,
    expected_label: str,
    request: pytest.FixtureRequest,
):
    """Score each BI-RADS example from the model card above 0.5 for its label."""
    pytest.importorskip("monai")
    density_scores = classify_breast_density(request.getfixturevalue(image))

    assert tuple(density_scores) == _LABELS
    assert all(
        isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
        for score in density_scores.values()
    )
    assert density_scores[expected_label] > 0.5
    return density_scores
