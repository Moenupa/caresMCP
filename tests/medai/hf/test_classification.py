"""Exercise standard pretrained classifiers on real images from public URLs."""

import math

import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.medai.hf import hf_classifier


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_id", "class_count"),
    [
        pytest.param("google/mobilenet_v2_1.0_224", 1001, id="mobilenet-v2"),
        pytest.param("microsoft/resnet-18", 1000, id="resnet-18"),
    ],
)
@skip_on_exception(ImportError, OSError)
@save_returns("hf_classifier", unpack=True)
def test_hf_classifier(sample: tuple[bytes, set[str]], model_id: str, class_count: int):
    """Predict cat labels for a real photo in both scoring modes."""
    image, expected_labels = sample
    outputs = {
        mode: hf_classifier(image, model_id, score_mode=mode)
        for mode in ("sigmoid", "softmax")
    }

    for mode, scores in outputs.items():
        context = f"{model_id}, {mode}"
        assert len(scores) == class_count, context
        assert all(label for label in scores), context
        assert all(
            math.isfinite(score) and 0 <= score <= 1 for score in scores.values()
        ), context
        top_labels = sorted(scores, key=scores.__getitem__, reverse=True)[:5]
        assert top_labels[0] in expected_labels, f"{context}: got {top_labels}"
        assert expected_labels <= set(top_labels), f"{context}: got {top_labels}"

    assert outputs["sigmoid"].keys() == outputs["softmax"].keys(), model_id
    assert sum(outputs["softmax"].values()) == pytest.approx(1.0, rel=1e-2)
    assert sum(outputs["sigmoid"].values()) > 1.0, model_id
    assert hf_classifier(image, model_id) == pytest.approx(outputs["sigmoid"])
    return outputs
