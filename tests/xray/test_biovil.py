"""Exercise official BioViL image encoders on real NIH radiographs."""

from typing import get_args

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.xray.biovil import BioViLModel, biovil_image_embed

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("model_id", get_args(BioViLModel))
@save_returns("biovil")
def test_biovil_image_embed(cxr_image: bytes, model_id: BioViLModel):
    """Check the normalized projected vector from each official checkpoint."""
    result = biovil_image_embed(cxr_image, model_id=model_id)
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
    single = biovil_image_embed(cxr_image_followup)
    temporal = biovil_image_embed(cxr_image_followup, prior_image=cxr_image)
    for result in (single, temporal):
        vector = np.asarray(result["embedding"])
        assert vector.shape == (128,)
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert single["prior_image_used"] is False
    assert temporal["prior_image_used"] is True
    assert not np.allclose(single["embedding"], temporal["embedding"])
    return {"current_only": single, "current_with_prior": temporal}
