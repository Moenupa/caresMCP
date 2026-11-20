"""Reproduce the released CT classifier's sample labels.

The checkpoint card
(https://huggingface.co/oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans)
publishes one negative (X) and one positive (Y) sample JPEG and states that X
marks a negative cancer image and Y a positive image. The card's 0.8852 test
accuracy is measured on an unpublished NLST test split, so the released
samples pin the author's per-image claim rather than that aggregate metric.

The sukhmani1303 checkpoint publishes a 97.58% test accuracy on a random
IQ-OTH/NCCD split whose membership is not released, and the repository
contains no sample image, so that score and its inputs cannot be reproduced.
"""

import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.ct.huggingface import classify_ct_slice

pytestmark = pytest.mark.slow
MODEL_ID = "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans"
RELEASED_SAMPLES = {
    "ct_released_negative": "negative",
    "ct_released_positive": "positive",
}


@pytest.mark.parametrize(("sample", "expected_label"), RELEASED_SAMPLES.items())
@skip_on_exception(ImportError, OSError)
@save_returns("ct_hf")
def test_released_sample_labels(
    sample: str,
    expected_label: str,
    request: pytest.FixtureRequest,
):
    """Classify each released sample as its author-labeled class."""
    result = classify_ct_slice(request.getfixturevalue(sample), model_id=MODEL_ID)

    scores = result["scores"]
    assert result["model_id"] == MODEL_ID
    assert result["task"] == "lung_cancer_classification"
    assert set(scores) == {"negative", "positive"}
    assert max(scores, key=lambda label: scores[label]) == expected_label
    assert scores[expected_label] > 0.5
    return result
