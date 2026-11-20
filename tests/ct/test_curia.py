"""Compare Curia's CT heads with the released Hub inference pipeline.

The model card reports 98.1% on CuriaBench Anatomy Recognition
(https://huggingface.co/raidium/curia). CuriaBench is gated, and its anatomy
head expects benchmark volumes and masks, so the cohort score cannot be
reproduced from the released inputs. Instead, every supported head is checked
against the released ``AutoImageProcessor``/``AutoModelForImageClassification``
pipeline on the pydicom project's CQ500 slice, which already carries the axial
PL orientation the model card requires.
"""

import functools
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.constants import DEVICE
from caresmcp.ct.curia import (
    CuriaCTTask,
    classify_ct_slice_with_curia,
)

pytestmark = pytest.mark.slow
MODEL_ID = "raidium/curia"


@functools.cache
def _released_processor():
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False
    )


@functools.cache
def _released_head(head: str):
    from transformers import AutoModelForImageClassification

    return (
        AutoModelForImageClassification.from_pretrained(
            MODEL_ID, subfolder=head, trust_remote_code=True
        )
        .eval()
        .to(DEVICE)
    )


@functools.cache
def _released_labels() -> dict[str, dict[int, str]]:
    from huggingface_hub import hf_hub_download

    document = json.loads(
        Path(hf_hub_download(MODEL_ID, "id_to_labels.json")).read_text()
    )
    return {
        head: {int(index): label for index, label in labels.items()}
        for head, labels in document.items()
    }


@pytest.mark.parametrize(
    "classification_task,head",
    [
        ("ct_anatomy_classification", "anatomy-ct"),
        ("covid_pneumonia_classification", "covidx-ct"),
        ("intracranial_hemorrhage_classification", "ich"),
    ],
)
@skip_on_exception(ImportError, OSError)
@save_returns("ct_curia")
def test_classify_dicom(
    ct_dicom: bytes,
    classification_task: CuriaCTTask,
    head: str,
):
    """Reproduce the released head's softmax scores for the same slice."""
    import pydicom
    import torch
    from pydicom.pixels import apply_modality_lut

    result = classify_ct_slice_with_curia(
        ct_dicom,
        classification_task=classification_task,
    )

    assert result["model_id"] == MODEL_ID
    assert result["head"] == head
    assert result["task"] == classification_task
    assert result["plane"] == "axial"
    assert result["prediction_level"] == "slice"
    assert result["intensity_units"] == "hu"
    assert result["score_mode"] == "softmax"

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    orientation = np.abs(np.asarray(dataset.ImageOrientationPatient).reshape(2, 3))
    # Axial PL: rows increase posteriorly and columns increase leftwards.
    np.testing.assert_allclose(orientation, [[1, 0, 0], [0, 1, 0]], atol=1e-3)
    hu = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32)

    inputs = _released_processor()(images=hu, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = _released_head(head)(**inputs)["logits"]
    probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
    labels = _released_labels()[head]
    released = {
        labels[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }

    scores = result["scores"]
    assert set(scores) == set(released)
    for label, probability in released.items():
        assert scores[label] == pytest.approx(probability, abs=1e-4)
    assert max(scores, key=lambda label: scores[label]) == max(
        released, key=lambda label: released[label]
    )
    return result


@skip_on_exception(ImportError, OSError)
def test_non_axial_plane_rejected_for_slice_heads(ct_dicom: bytes) -> None:
    """Only the anatomy head accepts coronal or sagittal input."""
    with pytest.raises(ValueError, match="requires axial CT slices"):
        classify_ct_slice_with_curia(
            ct_dicom,
            classification_task="covid_pneumonia_classification",
            anatomical_plane="coronal",
        )
