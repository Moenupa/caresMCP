"""Verify the unified encoder and generic matcher with real NIH radiographs."""

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.xray.embeddings import xray_image_embed, xray_image_similarity

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("pooling", ["cls", "mean"])
@save_returns("cxr_encoder", unpack=True)
def test_cxr_encoder(cxr_image: bytes, cxr_image_followup: bytes, pooling):
    """Rank two real images and reject genuinely different pooling representations."""
    pytest.importorskip("transformers")
    embedding = xray_image_embed(cxr_image, pooling=pooling)
    followup = xray_image_embed(cxr_image_followup, pooling=pooling)
    assert embedding["model_id"] == "microsoft/rad-dino"
    assert embedding["pooling"] == pooling
    vector = np.asarray(embedding["embedding"])
    assert vector.shape == (768,)
    assert np.isfinite(vector).all() and np.linalg.norm(vector) > 0
    return {"embedding": embedding, "followup": followup}


@save_returns("cxr_similarity")
def test_cxr_similarity(cxr_image: bytes, cxr_image_followup: bytes):
    """Compare real radiographs against their independently extracted embeddings."""
    embedding = xray_image_embed(cxr_image)
    followup = xray_image_embed(cxr_image_followup)
    vector = np.asarray(embedding["embedding"])
    similarity = xray_image_similarity(cxr_image, cxr_image_followup)
    other_vector = np.asarray(followup["embedding"])
    expected = np.dot(vector, other_vector) / (
        np.linalg.norm(vector) * np.linalg.norm(other_vector)
    )
    assert similarity["cosine_similarity"] == pytest.approx(expected, abs=1e-5)
    return similarity
