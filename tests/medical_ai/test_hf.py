"""Exercise standard pretrained classifiers on real images from public URLs."""

import math
from urllib.request import urlopen

import pytest

from caresmcp.medical_ai.hf import hf_classifier


@pytest.fixture(scope="module", autouse=True)
def require_hf_dependencies() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


@pytest.fixture(
    scope="module",
    params=[
        (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
            {"tabby, tabby cat", "tiger cat", "Egyptian cat"},
        ),
        (
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            {"Samoyed, Samoyede"},
        ),
    ],
    ids=["cats", "samoyed"],
)
def sample(request: pytest.FixtureRequest) -> tuple[bytes, set[str]]:
    """Download each real sample once and provide its expected ImageNet labels."""
    url, expected_labels = request.param
    with urlopen(url, timeout=30) as response:
        return response.read(), expected_labels


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_id", "class_count"),
    [
        pytest.param("google/mobilenet_v2_1.0_224", 1001, id="mobilenet-v2"),
        pytest.param("microsoft/resnet-18", 1000, id="resnet-18"),
    ],
)
def test_hf_classifier(
    sample: tuple[bytes, set[str]], model_id: str, class_count: int
) -> None:
    """Check recognizable predictions and both scoring modes with real weights."""
    image, expected_labels = sample
    outputs = {
        mode: hf_classifier(image, model_id, score_mode=mode)
        for mode in ("sigmoid", "softmax")
    }

    for mode, scores in outputs.items():
        context = f"{model_id}, {mode}"
        assert isinstance(scores, dict), context
        assert len(scores) == class_count, context
        assert all(isinstance(label, str) and label for label in scores), context
        assert all(
            isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
            for score in scores.values()
        ), context
        top_labels = sorted(scores, key=scores.__getitem__, reverse=True)[:5]
        assert expected_labels.intersection(top_labels), (
            f"{context}: expected {expected_labels} in top five, got {top_labels}"
        )

    assert outputs["sigmoid"].keys() == outputs["softmax"].keys(), model_id
    assert sum(outputs["softmax"].values()) == pytest.approx(1.0, abs=1e-5)
    assert sum(outputs["sigmoid"].values()) > 1.0, model_id
    assert hf_classifier(image, model_id) == pytest.approx(outputs["sigmoid"])
