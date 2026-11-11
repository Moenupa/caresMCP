"""Exercise Curia's public CT interface with real CT samples."""

import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.ct.curia import (
    ClassifyCtSliceWithCuriaClassificationTaskArg,
    classify_ct_slice_with_curia,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "classification_task,expected_head",
    [
        ("ct_anatomy_classification", "anatomy-ct"),
        ("covid_pneumonia_classification", "covidx-ct"),
        ("intracranial_hemorrhage_classification", "ich"),
    ],
)
@save_returns("ct_curia")
def test_classify_dicom(
    ct_lung_cancer_dicom: bytes,
    classification_task: ClassifyCtSliceWithCuriaClassificationTaskArg,
    expected_head: str,
):
    for package in ("torch", "transformers", "huggingface_hub"):
        pytest.importorskip(package)

    result = classify_ct_slice_with_curia(
        ct_lung_cancer_dicom,
        classification_task=classification_task,
    )

    assert result["model_id"] == "raidium/curia"
    assert result["head"] == expected_head
    assert result["task"] == classification_task
    assert result["plane"] == "axial"
    assert result["prediction_level"] == "slice"
    assert result["intensity_units"] == "hu"
    assert result["score_mode"] == "softmax"

    assert all(isinstance(label, str) and label for label in result["scores"])
    scores = list(result["scores"].values())
    assert len(scores) >= 2
    assert all(0 <= score <= 1 for score in scores)
    assert sum(scores) == pytest.approx(1, rel=1e-2)
    return result
