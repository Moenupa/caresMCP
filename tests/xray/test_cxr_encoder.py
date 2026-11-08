from typing import get_args

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.xray.cxr_encoder import (
    CXRModel,
    cxr_encoder_image_embed,
    cxr_encoder_retrieve_similar_reports,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("model_id", get_args(CXRModel))
@pytest.mark.parametrize("pooling", ["cls", "mean"])
@save_returns("cxr_encoder", unpack=True)
def test_cxr_encoder(cxr_image: bytes, model_id: CXRModel, pooling):
    """Check finite embeddings and retrieval of the image's own paired caption."""
    pytest.importorskip("transformers")
    embedding = cxr_encoder_image_embed(cxr_image, model_id=model_id, pooling=pooling)
    assert embedding["model_id"] == model_id
    assert embedding["pooling"] == pooling
    vector = np.asarray(embedding["embedding"])
    assert vector.ndim == 1 and vector.size > 0
    assert np.isfinite(vector).all() and np.linalg.norm(vector) > 0
    report = "NIH ChestX-ray14 example image 00000003_000."
    retrieval = cxr_encoder_retrieve_similar_reports(
        cxr_image,
        reference_reports=[report],
        reference_image_embeddings=[embedding["embedding"]],
        model_id=model_id,
        reference_pooling=pooling,
    )
    assert len(retrieval["matches"]) == 1
    assert retrieval["matches"][0]["report"] == report
    assert retrieval["matches"][0]["cosine_similarity"] == pytest.approx(1, abs=1e-5)
    return {"embedding": embedding, "retrieval": retrieval}
