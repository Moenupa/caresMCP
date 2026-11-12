"""Run the breast density classifier on real mammograms from the model card."""

import math

import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.mammography.density import _LABELS, classify_breast_density


@pytest.fixture(scope="module")
def density_scores(mlo_scattered_density: bytes) -> dict[str, float]:
    """Reuse one inference across the byte and path input tests."""
    pytest.importorskip("monai")
    return classify_breast_density(mlo_scattered_density)


@pytest.mark.slow
@save_returns("mammography_density")
def test_classify_breast_density(density_scores: dict[str, float]):
    """Score the real right MLO view as scattered fibroglandular density (B)."""
    assert tuple(density_scores) == _LABELS
    assert all(
        isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
        for score in density_scores.values()
    )
    assert density_scores[_LABELS[1]] > 0.5
    assert density_scores[_LABELS[2]] > 0.5
    return density_scores


@pytest.mark.slow
@pytest.mark.parametrize("image", ["a_sample", "b_sample", "c_sample", "d_sample"])
@save_returns("mammography_density")
def test_classify_breast_density_model_card_samples(
    image: str,
    request: pytest.FixtureRequest,
):
    """Score the four BI-RADS examples released with the model card."""
    pytest.importorskip("monai")
    density_scores = classify_breast_density(request.getfixturevalue(image))

    assert tuple(density_scores) == _LABELS
    assert all(
        isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
        for score in density_scores.values()
    )
    assert density_scores[_LABELS[0]] < 0.5
    assert density_scores[_LABELS[1]] > 0.5
    assert density_scores[_LABELS[2]] > 0.5
    assert density_scores[_LABELS[3]] < 0.5
    return density_scores
