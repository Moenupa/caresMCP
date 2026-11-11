"""Exercise DALE-CT-2S through its public CT interface."""

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.ct.dale_ct_2s import dale_ct_2s_image_embed


@pytest.mark.slow
@pytest.mark.parametrize("image", ["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"])
@save_returns("ct_dale_ct_2s")
def test_dicom_embedding(
    image: str,
    request: pytest.FixtureRequest,
):
    for package in ("torch", "timm", "transformers"):
        pytest.importorskip(package)
    result = dale_ct_2s_image_embed(request.getfixturevalue(image))

    assert result["model_id"] == "hf-hub:Kentucky-Open-Science/DALE-CT-2S"
    assert result["embedding_type"] == "global"
    assert len(result["image_shape_hw"]) == 2
    assert all(size > 0 and size % 16 == 0 for size in result["image_shape_hw"])
    embedding = np.asarray(result["embedding"])
    assert embedding.ndim == 1
    assert embedding.size > 0
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) > 0
    return result
