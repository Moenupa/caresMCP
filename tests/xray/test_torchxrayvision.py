"""Exercise TorchXRayVision tools with a real NIH chest radiograph."""

from io import BytesIO
from typing import Literal, get_args

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.api.models import get_torchxrayvision_model
from caresmcp.xray.torchxrayvision import (
    ANATOMICAL_TARGETS,
    ChestXrayWeights,
    torchxrayvision_classify,
    torchxrayvision_localize,
    torchxrayvision_segment_anatomy,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", autouse=True)
def require_torchxrayvision():
    """Skip integration tests when optional X-ray dependencies are absent."""
    for package in ("torch", "torchvision", "torchxrayvision"):
        pytest.importorskip(package)


@pytest.mark.parametrize("weights", get_args(ChestXrayWeights))
@save_returns("torchxrayvision")
def test_torchxrayvision_classify(cxr_image: bytes, weights: ChestXrayWeights):
    """Check pathology names, score ranges, and rounding for every checkpoint."""
    result = torchxrayvision_classify(cxr_image, weights=weights)

    pathology_scores = result["scores"]
    assert pathology_scores
    assert all(isinstance(name, str) and name for name in pathology_scores)
    assert all(
        isinstance(score, float) and 0 <= score <= 1
        for score in pathology_scores.values()
    )
    assert all(score == round(score, 4) for score in pathology_scores.values())
    return result


@pytest.mark.parametrize("output", ["legend", "label_mask"])
@save_returns("torchxrayvision")
def test_torchxrayvision_segment_anatomy(
    cxr_image: bytes, output: Literal["legend", "label_mask"]
):
    """Check the mask geometry, palette, background transparency, and label IDs."""
    result = torchxrayvision_segment_anatomy(cxr_image, output=output)
    assert tuple(get_torchxrayvision_model("anatomy").targets) == ANATOMICAL_TARGETS, (
        "Anatomical model target order must match mask IDs"
    )

    with Image.open(BytesIO(result)) as mask:
        assert mask.format == "PNG"
        assert mask.mode == "P"
        assert mask.height == 512
        assert mask.width == 512 if output == "label_mask" else mask.width > 512
        assert mask.info["transparency"] == 0
        labels = np.asarray(mask)[:, :512]
        assert set(np.unique(labels)) <= set(range(len(ANATOMICAL_TARGETS) + 1))
        assert np.any(labels > 0), (
            "Expected anatomical structures in the real radiograph"
        )
    return result


@save_returns("torchxrayvision")
def test_torchxrayvision_segment_anatomy_region(cxr_image: bytes):
    """Check that selecting a lung returns a binary mask without a legend."""
    result = torchxrayvision_segment_anatomy(cxr_image, region="Left Lung")
    with Image.open(BytesIO(result)) as mask:
        assert mask.format == "PNG"
        assert mask.mode == "L"
        assert mask.size == (512, 512)
        assert set(np.unique(np.asarray(mask))) <= {0, 255}
    return result


@pytest.mark.parametrize(
    ("weights", "size"), [("densenet121-res224-all", 224), ("resnet50-res512-all", 512)]
)
@pytest.mark.parametrize("output", ["overlay", "saliency"])
@save_returns("torchxrayvision")
def test_torchxrayvision_localize(
    cxr_image: bytes,
    weights: ChestXrayWeights,
    size: int,
    output: Literal["overlay", "saliency"],
):
    """Check Grad-CAM PNGs from both supported feature-map architectures."""
    import torch

    model = get_torchxrayvision_model(weights)
    assert "Hernia" in model.pathologies, f"{weights} must expose the requested label"
    target_layer = (
        model.model.layer4
        if weights.startswith("resnet")
        else model.features.denseblock4
    )
    assert isinstance(target_layer, torch.nn.Module)
    activations = []
    hook = target_layer.register_forward_hook(
        lambda _module, _inputs, features: activations.append(features)
    )
    try:
        result = torchxrayvision_localize(
            cxr_image, label="Hernia", weights=weights, output=output
        )
    finally:
        hook.remove()
    assert len(activations) == 1, (
        f"{weights}: expected one final feature map, captured {len(activations)}"
    )
    with Image.open(BytesIO(result)) as image:
        assert image.format == "PNG"
        assert image.size == (size, size)
        assert image.mode == ("RGB" if output == "overlay" else "L")
        pixels = np.asarray(image)
        assert pixels.dtype == np.uint8
        if output == "saliency":
            assert pixels.max() == 255 or not pixels.any()
    return result


@save_returns("torchxrayvision", unpack=True)
def test_overlay_saliency(cxr_image: bytes):
    """Exercise the generic overlay with a real saliency map and aligned image."""
    from caresmcp.api.imageio import encode_as
    from caresmcp.tools.saliency import overlay_saliency

    saliency = torchxrayvision_localize(cxr_image, "Hernia", output="saliency")
    with Image.open(BytesIO(cxr_image)) as source:
        base = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    image = encode_as(base)
    overlay = overlay_saliency(image, saliency)
    transparent = overlay_saliency(image, saliency, opacity=0)
    with (
        Image.open(BytesIO(overlay)) as output,
        Image.open(BytesIO(transparent)) as zero,
    ):
        assert output.size == base.size
        assert output.mode == "RGB"
        assert np.array_equal(np.asarray(zero), np.asarray(base))
    with pytest.raises(ValueError, match="dimensions must match"):
        overlay_saliency(cxr_image, saliency)
    return {"saliency": saliency, "aligned_source": image, "overlay": overlay}
