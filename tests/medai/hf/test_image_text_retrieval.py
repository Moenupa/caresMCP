"""Exercise dual-encoder image-text retrieval on a real chest radiograph."""

import math
from typing import get_args

import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.medai.hf import ImageTextRetrievalModel, hf_image_text_retrieval

TEXTS = [
    "a photo of a cat",
    "a chest x-ray showing a hernia",
    "an MRI of the brain",
]
CHEST_TEXT_INDEX = 1


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param(model_id, id=model_id.rsplit("/", 1)[-1])
        for model_id in get_args(ImageTextRetrievalModel)
    ],
)
@skip_on_exception(ImportError, OSError)
@save_returns("hf_image_text_retrieval")
def test_hf_image_text_retrieval(
    hernia_image: bytes, model_id: ImageTextRetrievalModel
):
    """Rank real candidates and retain input order and cosine score bounds."""
    result = hf_image_text_retrieval(hernia_image, TEXTS, model_id=model_id)
    assert result["model_id"] == model_id
    assert result["score_type"] == "cosine_similarity"
    assert [match["text"] for match in result["matches"]] == TEXTS
    scores = [match["score"] for match in result["matches"]]
    assert all(math.isfinite(score) and -1 <= score <= 1 for score in scores)
    # The radiograph must outrank the off-modality candidates in every encoder.
    assert scores.index(max(scores)) == CHEST_TEXT_INDEX
    return result


@pytest.mark.slow
@skip_on_exception(ImportError, OSError)
@save_returns("hf_image_text_retrieval")
def test_hf_image_text_retrieval_default_model(hernia_image: bytes):
    """Select the gated medical encoder when model_id is omitted."""
    result = hf_image_text_retrieval(hernia_image, [TEXTS[CHEST_TEXT_INDEX]])
    assert result["model_id"] == "google/medsiglip-448"
    assert [match["text"] for match in result["matches"]] == [TEXTS[CHEST_TEXT_INDEX]]
    return result
