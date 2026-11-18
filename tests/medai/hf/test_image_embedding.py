"""Exercise image-encoder embeddings on real images from public URLs."""

import math
from typing import Literal

import pytest

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.medai.hf import hf_image_embed


@pytest.mark.slow
@pytest.mark.parametrize("pooling", ["cls", "mean"])
@skip_on_import_error
@save_returns("hf_image_embed")
def test_hf_image_embed(
    sample: tuple[bytes, set[str]], pooling: Literal["cls", "mean"]
):
    """Extract finite image embeddings using both token pooling options."""
    image, _ = sample
    result = hf_image_embed(image=image, pooling=pooling)
    assert result["model_id"] == "facebook/dinov2-base"
    assert result["pooling"] == pooling
    assert len(result["embedding"]) == 768
    assert all(math.isfinite(value) for value in result["embedding"])
    assert any(value != 0 for value in result["embedding"])
    return result
