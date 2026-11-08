"""Exercise CT Hugging Face tools with real DICOM samples."""

from typing import Literal

import numpy as np
import pytest

from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.ct.huggingface import (
    CTClassifier,
    classify_ct_slice,
    ct_slice_image_embed,
)

pytestmark = pytest.mark.slow
SUITE = "ct_hf"


@pytest.fixture(scope="module")
def image(request: pytest.FixtureRequest) -> bytes:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "image", ["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"], indirect=True
)
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_ct_slice_image_embed(
    image: bytes, pooling: Literal["cls", "mean"], request: pytest.FixtureRequest
) -> None:
    result = ct_slice_image_embed(image, pooling=pooling)
    save_results(result, CACHE_ROOT / SUITE / request.node.name)

    embedding = np.asarray(result["embedding"])
    assert embedding.shape == (768,)
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) > 0


@pytest.mark.parametrize(
    "image", ["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"], indirect=True
)
@pytest.mark.parametrize(
    "model_id, labels",
    [
        pytest.param(
            "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans",
            {"negative", "positive"},
            id="swin",
        ),
        pytest.param(
            "sukhmani1303/lung-cancer-vit-model",
            {"Cancer", "Normal"},
            marks=pytest.mark.cuda,
            id="vit",
        ),
    ],
)
def test_classify_ct_slice(
    image: bytes,
    model_id: CTClassifier,
    labels: set[str],
    request: pytest.FixtureRequest,
) -> None:
    if model_id == "sukhmani1303/lung-cancer-vit-model":
        pytest.importorskip("timm")

    result = classify_ct_slice(image, model_id=model_id)
    save_results(result, CACHE_ROOT / SUITE / request.node.name)

    assert set(result["scores"]) == labels
    scores = list(result["scores"].values())
    assert all(0 <= score <= 1 for score in scores)
    assert sum(scores) == pytest.approx(1)
