"""Exercise official BioViL image encoders on real NIH radiographs."""

from typing import get_args

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.xray.biovil import BioViLModel
from caresmcp.xray.embeddings import xray_image_embed

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("model_id", get_args(BioViLModel))
@save_returns("biovil")
def test_biovil_image_embed(cxr_image: bytes, model_id: BioViLModel):
    """Check the normalized projected vector from each official checkpoint."""
    result = xray_image_embed(cxr_image, model_id=model_id)
    vector = np.asarray(result["embedding"])
    assert vector.shape == (128,)
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert result["model_id"] == model_id
    assert result["prior_image_used"] is False
    return result


@save_returns("biovil", unpack=True)
def test_biovil_temporal_embed(cxr_image: bytes, cxr_image_followup: bytes):
    """Exercise the paired-study pathway with two NIH images of the same patient."""
    single = xray_image_embed(
        cxr_image_followup, model_id="microsoft/BiomedVLP-BioViL-T"
    )
    temporal = xray_image_embed(
        cxr_image_followup,
        model_id="microsoft/BiomedVLP-BioViL-T",
        prior_image=cxr_image,
    )
    for result in (single, temporal):
        vector = np.asarray(result["embedding"])
        assert vector.shape == (128,)
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert single["prior_image_used"] is False
    assert temporal["prior_image_used"] is True
    assert not np.allclose(single["embedding"], temporal["embedding"])
    return {"current_only": single, "current_with_prior": temporal}


@pytest.mark.parametrize("model_id", get_args(BioViLModel))
@save_returns("biovil")
def test_biovil_ground_phrases(cxr_image: bytes, model_id: BioViLModel):
    """Inspect phrase maps and verify explicit crop coverage on a real CXR."""
    from caresmcp.xray.biovil import biovil_ground_phrases

    result = biovil_ground_phrases(
        cxr_image, ["Left lung", "Right lung", "Heart"], model_id=model_id
    )
    assert result["model_id"] == model_id
    assert result["outside_coverage"] == "unobserved"
    x0, y0, x1, y1 = result["coverage_xyxy_original_pixels"]
    assert 0 < x0 < x1 < result["image_width"]
    assert 0 < y0 < y1 < result["image_height"]
    for phrase in result["phrases"]:
        values = np.asarray(phrase["patch_similarity_map"])
        assert values.ndim == 2 and min(values.shape) > 1
        assert np.isfinite(values).all()
        assert values.min() >= -1.00001 and values.max() <= 1.00001
        assert -1.00001 <= phrase["global_cosine_similarity"] <= 1.00001
    scores = [
        result["phrases"][index]["global_cosine_similarity"]
        for index in result["ranked_phrase_indices"]
    ]
    assert scores == sorted(scores, reverse=True)
    return result


@save_returns("biovil", unpack=True)
def test_biovil_temporal_grounding(cxr_image: bytes, cxr_image_followup: bytes):
    """Check paired-image grounding without assigning progression categories."""
    from caresmcp.xray.biovil import biovil_ground_phrases

    current = biovil_ground_phrases(cxr_image_followup, ["Left lung", "Right lung"])
    paired = biovil_ground_phrases(
        cxr_image_followup, ["Left lung", "Right lung"], prior_image=cxr_image
    )
    assert current["prior_image_used"] is False
    assert paired["prior_image_used"] is True
    assert (
        current["coverage_xyxy_original_pixels"]
        == paired["coverage_xyxy_original_pixels"]
    )
    current_scores = [
        phrase["global_cosine_similarity"] for phrase in current["phrases"]
    ]
    paired_scores = [phrase["global_cosine_similarity"] for phrase in paired["phrases"]]
    assert np.isfinite(paired_scores).all()
    assert not np.allclose(current_scores, paired_scores)
    return {"current_only": current, "current_with_prior": paired}
