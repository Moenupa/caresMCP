"""Exercise CT Hugging Face tools with real DICOM samples."""

import pytest

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.ct.huggingface import CTClassifier, classify_ct_slice

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("image", ["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"])
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
@skip_on_import_error
@save_returns("ct_hf")
def test_classify_ct_slice(
    image: str,
    model_id: CTClassifier,
    labels: set[str],
    request: pytest.FixtureRequest,
):
    result = classify_ct_slice(request.getfixturevalue(image), model_id=model_id)

    assert set(result["scores"]) == labels
    scores = list(result["scores"].values())
    assert all(0 <= score <= 1 for score in scores)
    assert sum(scores) == pytest.approx(1)
    return result
