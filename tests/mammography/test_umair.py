import math

import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.mammography.umair import classify_mammogram_patch


@pytest.mark.slow
@pytest.mark.parametrize("image", ["mlo_scattered_density", "b_sample", "c_sample"])
@save_returns("mammography_classification")
def test_classify_mammogram_patch(image: str, request: pytest.FixtureRequest):
    """Return normalized benign, malignant, and normal scores for real images."""
    pytest.importorskip("torchvision")
    scores = classify_mammogram_patch(request.getfixturevalue(image))

    assert tuple(scores) == ("benign", "malignant", "normal")
    assert all(
        isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
        for score in scores.values()
    )
    assert math.isclose(sum(scores.values()), 1.0, abs_tol=1e-3)
    return scores
