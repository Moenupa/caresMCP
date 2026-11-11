"""Exercise SKELEX masked reconstruction on real pediatric wrist radiographs."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.api.io import encode_as
from caresmcp.api.models import HFModelType, get_hf_model_and_processor
from caresmcp.xray.skelex import skelex_reconstruction_error

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


@save_returns("skelex", unpack=True)
def test_skelex_reconstruction_error(wrist_example):
    """Check mask coverage, error scale and reproducible multi-pass sampling."""
    pytest.importorskip("transformers")
    import torch

    model, _ = get_hf_model_and_processor(
        "skhoha/SKELEX", HFModelType.IMAGE_PRETRAINING, use_fast=False
    )
    assert model.config.norm_pix_loss is False
    assert model.config.mask_ratio == 0.75
    image, _ = wrist_example
    result = skelex_reconstruction_error(image, passes=3, seed=42)
    rng_state = torch.random.get_rng_state().clone()
    repeated = skelex_reconstruction_error(image, passes=3, seed=42)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert result == repeated
    with Image.open(BytesIO(result)) as heatmap:
        assert heatmap.size == (224, 224)
        assert heatmap.mode == "RGBA"
        pixels = np.asarray(heatmap)
        assert pixels[..., 0].max() == 255
        assert np.all(pixels[..., 0] == pixels[..., 1])
        assert set(np.unique(pixels[..., 3])) <= {0, 255}
        assert heatmap.info["model_id"] == "skhoha/SKELEX"
        assert heatmap.info["mask_ratio"] == "0.75"
        assert 0.75 <= float(heatmap.info["observed_pixel_fraction"]) <= 1
        assert 0 < float(heatmap.info["observed_pixel_mean_squared_error"])
        metadata = dict(heatmap.info)
    return {"source": image, "heatmap": result, "metadata": metadata}


@save_returns("skelex_palette", unpack=True)
def test_skelex_palette_image(wrist_example):
    """A palette image and its RGB rendering should yield the same map."""
    pytest.importorskip("transformers")
    image, _ = wrist_example
    with Image.open(BytesIO(image)) as source:
        palette = source.convert("RGB").quantize(colors=64)
    palette_image = encode_as(palette)
    rgb_image = encode_as(palette.convert("RGB"))
    result = skelex_reconstruction_error(palette_image, passes=1, seed=42)
    assert result == skelex_reconstruction_error(rgb_image, passes=1, seed=42)
    return {"source": palette_image, "heatmap": result}
