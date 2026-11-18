"""Exercise dual-encoder image-text retrieval on a real chest radiograph."""

import math

import pytest

from caresmcp.api.decorators import (
    save_returns,
    skip_on_exception,
    skip_on_import_error,
)
from caresmcp.medai.hf import ImageTextRetrievalModel, hf_image_text_retrieval


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("openai/clip-vit-base-patch32", id="clip-vit-b32"),
        pytest.param("google/siglip-base-patch16-224", id="siglip-b16"),
    ],
)
@skip_on_import_error
@save_returns("hf_image_text_retrieval")
def test_hf_image_text_retrieval(
    hernia_image: bytes, model_id: ImageTextRetrievalModel
) -> dict:
    """Rank real candidate texts and retain input order and cosine score bounds."""
    texts = [
        "a chest x-ray showing a hernia",
        "a photo of a cat",
        "an MRI of the brain",
    ]
    result = hf_image_text_retrieval(hernia_image, texts, model_id=model_id)
    assert result["model_id"] == model_id
    assert result["score_type"] == "cosine_similarity"
    assert [match["text"] for match in result["matches"]] == texts
    scores = [match["score"] for match in result["matches"]]
    assert all(math.isfinite(score) and -1 <= score <= 1 for score in scores)
    return result


@pytest.mark.slow
@skip_on_exception(ImportError, OSError)
def test_hf_image_text_retrieval_medsiglip(hernia_image: bytes) -> None:
    """Exercise the default gated medical encoder when access has been granted."""
    result = hf_image_text_retrieval(hernia_image, ["a chest x-ray showing a hernia"])
    assert result["model_id"] == "google/medsiglip-448"
    assert len(result["matches"]) == 1


@skip_on_import_error
def test_hf_image_text_retrieval_requires_text(dummy_jpeg: bytes) -> None:
    """Reject empty candidates before loading a checkpoint."""
    with pytest.raises(ValueError, match="non-empty"):
        hf_image_text_retrieval(dummy_jpeg, [])
    with pytest.raises(ValueError, match="non-empty"):
        hf_image_text_retrieval(dummy_jpeg, ["   "])
