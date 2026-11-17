"""Exercise TorchXRayVision tools with a real NIH chest radiograph."""

from typing import Literal, get_args

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as
from caresmcp.api.models import get_torchxrayvision_model
from caresmcp.xray import torchxrayvision as xray_tools
from caresmcp.xray.torchxrayvision import (
    ANATOMICAL_TARGETS,
    ChestXrayWeights,
    torchxrayvision_classify,
    torchxrayvision_localize,
    torchxrayvision_segment_anatomy,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("weights", get_args(ChestXrayWeights))
@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_classify(hernia_image: bytes, weights: ChestXrayWeights):
    """Check pathology names, score ranges, and rounding for every checkpoint."""
    result = torchxrayvision_classify(hernia_image, weights=weights)

    pathology_scores = result["scores"]
    assert pathology_scores
    assert all(isinstance(name, str) and name for name in pathology_scores)
    available_scores = {
        name: score for name, score in pathology_scores.items() if score is not None
    }
    assert available_scores
    assert all(
        isinstance(score, float) and 0 <= score <= 1
        for score in available_scores.values()
    )
    if weights == "resnet50-res512-all":
        assert pathology_scores["Lung Lesion"] is None
        assert pathology_scores["Enlarged Cardiomediastinum"] is None
    else:
        assert all(score is not None for score in pathology_scores.values())

    # expected hernia
    if "Hernia" in pathology_scores:
        assert pathology_scores["Hernia"] > 0.50
    return result


@pytest.mark.parametrize("output", ["legend", "label_mask"])
@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_segment_anatomy(
    hernia_image: bytes, output: Literal["legend", "label_mask"]
):
    """Check the mask geometry, palette, background transparency, and label IDs."""
    result = torchxrayvision_segment_anatomy(hernia_image, output=output)
    assert tuple(get_torchxrayvision_model("anatomy").targets) == ANATOMICAL_TARGETS, (
        "Anatomical model target order must match mask IDs"
    )

    mask = decode_as(result)
    assert mask.format == "PNG"
    assert mask.mode == "P"
    assert mask.height == 512
    assert mask.width == 512 if output == "label_mask" else mask.width > 512
    assert mask.info["transparency"] == 0
    labels = np.asarray(mask)[:, :512]
    assert set(np.unique(labels)) <= set(range(len(ANATOMICAL_TARGETS) + 1))
    assert np.any(labels > 0), "Expected anatomical structures in the real radiograph"
    return result


@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_segment_anatomy_region(hernia_image: bytes):
    """Check that selecting a lung returns a binary mask without a legend."""
    result = torchxrayvision_segment_anatomy(hernia_image, region="Left Lung")
    mask = decode_as(result)
    assert mask.format == "PNG"
    assert mask.mode == "L"
    assert mask.size == (512, 512)
    assert set(np.unique(np.asarray(mask))) <= {0, 255}
    return result


@pytest.mark.parametrize(
    ("weights", "size"), [("densenet121-res224-all", 224), ("resnet50-res512-all", 512)]
)
@pytest.mark.parametrize("output", ["overlay", "saliency"])
@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_localize(
    hernia_image: bytes,
    weights: ChestXrayWeights,
    size: int,
    output: Literal["overlay", "saliency"],
    monkeypatch: pytest.MonkeyPatch,
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
    # Pass this exact instance to the tool so the hook observes its forward pass.
    monkeypatch.setattr(
        xray_tools, "get_torchxrayvision_model", lambda *_args, **_kwargs: model
    )
    activations = []
    hook = target_layer.register_forward_hook(
        lambda _module, _inputs, features: activations.append(features)
    )
    try:
        result = torchxrayvision_localize(
            hernia_image, label="Hernia", weights=weights, output=output
        )
    finally:
        hook.remove()
    assert len(activations) == 1, (
        f"{weights}: expected one final feature map, captured {len(activations)}"
    )
    image = decode_as(result)
    assert image.format == "PNG"
    assert image.size == (size, size)
    assert image.mode == ("RGB" if output == "overlay" else "L")
    pixels = np.asarray(image)
    assert pixels.dtype == np.uint8
    if output == "saliency":
        assert pixels.max() == 255 or not pixels.any()
    return result


@skip_on_import_error
def test_torchxrayvision_localize_unavailable_label(cxr_image: bytes):
    """Check labels without an operating threshold fail instead of rendering a blank map."""
    with pytest.raises(ValueError, match="no operating threshold"):
        torchxrayvision_localize(
            cxr_image, label="Lung Lesion", weights="resnet50-res512-all"
        )


@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_classify_dicom(chest_dicom: bytes):
    """Check DICOM input scores identically to its pixels supplied as a raster."""
    from io import BytesIO

    import pydicom
    from PIL import Image

    from caresmcp.api.io import encode_as

    dataset = pydicom.dcmread(BytesIO(chest_dicom))
    raster = encode_as(Image.fromarray(dataset.pixel_array, mode="L"))
    dicom_result = torchxrayvision_classify(chest_dicom)
    raster_result = torchxrayvision_classify(raster)
    assert dicom_result["scores"] == raster_result["scores"]
    assert any(score is not None for score in dicom_result["scores"].values())
    return dicom_result


@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_segment_anatomy_dicom(chest_dicom: bytes):
    """Check anatomy segmentation accepts a DICOM radiograph."""
    result = torchxrayvision_segment_anatomy(chest_dicom, region="Left Lung")
    mask = decode_as(result)
    assert mask.format == "PNG"
    assert mask.mode == "L"
    assert mask.size == (512, 512)
    assert np.any(np.asarray(mask) > 0), "Expected a lung in the real radiograph"
    return result


@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_localize_dicom(chest_dicom: bytes):
    """Check Grad-CAM saliency accepts a DICOM radiograph."""
    result = torchxrayvision_localize(chest_dicom, label="Effusion")
    image = decode_as(result)
    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (224, 224)
    assert np.asarray(image).any()
    return result
