"""Exercise TorchXRayVision tools with a real NIH chest radiograph.

The target Hernia score, masks, and saliency maps are checked against recorded
values from a deterministic CPU run, not only against generic shape and range
contracts.
"""

from io import BytesIO
from typing import Literal

import numpy as np
import pydicom
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as, encode_as
from caresmcp.api.metrics import mask_overlap
from caresmcp.api.models import get_torchxrayvision_model
from caresmcp.xray.torchxrayvision import (
    ANATOMICAL_TARGETS,
    ChestXrayWeights,
    torchxrayvision_classify,
    torchxrayvision_localize,
    torchxrayvision_segment_anatomy,
)

pytestmark = pytest.mark.slow

# Labels exposed by each checkpoint. Values that a checkpoint exposes but
# cannot score are listed in EXPECTED_UNAVAILABLE.
ALL_LABELS = frozenset(
    {
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    }
)
NIH_LABELS = frozenset(
    {
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
    }
)
EXPECTED_LABELS: dict[ChestXrayWeights, frozenset[str]] = {
    "densenet121-res224-all": ALL_LABELS,
    "densenet121-res224-nih": NIH_LABELS,
    "densenet121-res224-pc": NIH_LABELS | {"Fracture"},
    "resnet50-res512-all": ALL_LABELS,
}
EXPECTED_UNAVAILABLE: dict[ChestXrayWeights, frozenset[str]] = {
    "resnet50-res512-all": frozenset({"Lung Lesion", "Enlarged Cardiomediastinum"}),
}

# Recorded Hernia scores from a deterministic CPU run on the NIH radiograph.
# These are operating-point-normalized outputs, not probabilities: each
# checkpoint maps its own per-label threshold to 0.5, so values are not
# comparable across checkpoints. The nih-only checkpoint is weakest on Hernia
# (raw sigmoid ~0.07, mapped to 0.5371 by its 0.00044 threshold; NIH has 227
# positives in 112,120). The four selected checkpoints cover both released
# architectures and each distinct label schema while bounding slow downloads.
EXPECTED_HERNIA_SCORES: dict[ChestXrayWeights, float] = {
    "densenet121-res224-all": 0.9981,
    "densenet121-res224-nih": 0.5371,
    "densenet121-res224-pc": 0.9298,
    "resnet50-res512-all": 0.9637,
}


@pytest.mark.parametrize("weights", tuple(EXPECTED_HERNIA_SCORES.keys()))
@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_classify(hernia_image: bytes, weights: ChestXrayWeights):
    """Check the checkpoint, label set, unavailable labels, and Hernia score."""
    result = torchxrayvision_classify(hernia_image, weights=weights)

    assert result["model_id"] == weights
    assert result["score_mode"] == "sigmoid"
    scores = result["scores"]
    assert set(scores) == EXPECTED_LABELS[weights]
    unavailable = {name for name, score in scores.items() if score is None}
    assert unavailable == EXPECTED_UNAVAILABLE.get(weights, frozenset())
    available = {name: score for name, score in scores.items() if score is not None}
    assert available
    assert all(
        isinstance(score, float) and 0 <= score <= 1 for score in available.values()
    )

    # The study contains a hernia; every selected checkpoint must detect it and
    # reproduce the recorded score.
    hernia_score = scores["Hernia"]
    assert hernia_score is not None
    assert hernia_score == pytest.approx(EXPECTED_HERNIA_SCORES[weights], rel=1e-2)
    assert hernia_score > 0.5
    return result


@skip_on_import_error
def test_torchxrayvision_anatomy_target_order():
    """Check mask IDs follow the shared anatomy model's target order."""
    model = get_torchxrayvision_model("anatomy")
    assert tuple(model.targets) == ANATOMICAL_TARGETS


@pytest.mark.parametrize("output", ["legend", "label_mask"])
@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_segment_anatomy(
    hernia_image: bytes, output: Literal["legend", "label_mask"]
):
    """Check mask geometry, palette, background transparency, and label IDs."""
    result = torchxrayvision_segment_anatomy(hernia_image, output=output)
    mask = decode_as(result)
    assert mask.format == "PNG"
    assert mask.mode == "P"
    assert mask.height == 512
    if output == "label_mask":
        assert mask.size == (512, 512)
    else:
        assert mask.width > 512
    assert mask.info["transparency"] == 0
    palette = mask.getpalette()
    assert palette is not None and len(palette) == 768
    assert palette[:3] == [0, 0, 0], "Background must be black"

    labels = np.asarray(mask)[:, :512]
    assert set(np.unique(labels)) <= set(range(len(ANATOMICAL_TARGETS) + 1))
    core_targets = {
        "Left Lung",
        "Right Lung",
        "Heart",
        "Facies Diaphragmatica",
        "Spine",
    }
    core_ids = {ANATOMICAL_TARGETS.index(name) + 1 for name in core_targets}
    assert core_ids <= set(np.unique(labels)), (
        "Expected core anatomy in the real radiograph"
    )
    return result


@pytest.mark.parametrize("image_fixture", ["hernia_image", "chest_dicom"])
@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_segment_anatomy_region(
    request: pytest.FixtureRequest, image_fixture: str
):
    """Check an independently thresholded structure mask against the combined mask."""
    image = request.getfixturevalue(image_fixture)
    combined = np.asarray(
        decode_as(torchxrayvision_segment_anatomy(image, output="label_mask"))
    )
    result = torchxrayvision_segment_anatomy(image, region="Left Lung")
    mask = decode_as(result)
    assert mask.format == "PNG"
    assert mask.mode == "L"
    assert mask.size == (512, 512)
    region = np.asarray(mask)
    assert set(np.unique(region)) <= {0, 255}

    region_pixels = region > 0
    combined_left_lung = combined == ANATOMICAL_TARGETS.index("Left Lung") + 1
    assert combined_left_lung.any(), "Expected the left lung in the real radiograph"
    # Region masks threshold one structure independently, so they retain every
    # pixel that the combined highest-score mask assigned to that structure.
    assert np.all(region_pixels[combined_left_lung])
    dice = mask_overlap(region_pixels, combined_left_lung).dice
    assert dice is not None and dice > 0.8
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
):
    """Check Grad-CAM PNGs from both supported classifier architectures."""
    result = torchxrayvision_localize(
        hernia_image, label="Hernia", weights=weights, output=output
    )
    image = decode_as(result)
    assert image.format == "PNG"
    assert image.size == (size, size)
    assert image.mode == ("RGB" if output == "overlay" else "L")
    pixels = np.asarray(image)
    assert pixels.dtype == np.uint8
    assert pixels.any()
    if output == "overlay":
        # A plain grayscale radiograph would have equal color channels.
        assert not np.array_equal(pixels[..., 0], pixels[..., 1])
        assert not np.array_equal(pixels[..., 1], pixels[..., 2])
    else:
        # compute_gradcam scales the positive map to the full uint8 range.
        assert pixels.min() == 0
        assert pixels.max() == 255
    return result


@skip_on_import_error
@save_returns("torchxrayvision", unpack=True)
def test_torchxrayvision_localize_is_label_specific(hernia_image: bytes):
    """Check different findings produce different saliency for the same input."""
    results = {
        label: torchxrayvision_localize(
            hernia_image,
            label=label,
            weights="densenet121-res224-all",
            output="saliency",
        )
        for label in ("Hernia", "Effusion")
    }
    hernia = np.asarray(decode_as(results["Hernia"]), dtype=np.int16)
    effusion = np.asarray(decode_as(results["Effusion"]), dtype=np.int16)
    assert not np.array_equal(hernia, effusion)
    assert np.abs(hernia - effusion).mean() > 10
    return results


@skip_on_import_error
def test_torchxrayvision_localize_unavailable_label(hernia_image: bytes):
    """Check labels without an operating threshold fail instead of rendering a blank map."""
    with pytest.raises(ValueError, match="no operating threshold"):
        torchxrayvision_localize(
            hernia_image, label="Lung Lesion", weights="resnet50-res512-all"
        )


@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_classify_dicom(chest_dicom: bytes):
    """Check DICOM input scores identically to its stored pixels supplied as PNG."""
    dataset = pydicom.dcmread(BytesIO(chest_dicom))
    pixels = np.asarray(dataset.pixel_array, dtype=np.uint8)
    raster = encode_as(Image.fromarray(pixels))
    dicom_result = torchxrayvision_classify(chest_dicom)
    raster_result = torchxrayvision_classify(raster)
    assert dicom_result["model_id"] == "densenet121-res224-all"
    assert dicom_result["score_mode"] == "sigmoid"
    assert dicom_result == raster_result
    assert any(score is not None for score in dicom_result["scores"].values())
    return dicom_result


@skip_on_import_error
@save_returns("torchxrayvision")
def test_torchxrayvision_localize_dicom(chest_dicom: bytes):
    """Check Grad-CAM saliency accepts a DICOM radiograph."""
    result = torchxrayvision_localize(chest_dicom, label="Effusion")
    image = decode_as(result)
    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (224, 224)
    pixels = np.asarray(image)
    assert not np.array_equal(pixels[..., 0], pixels[..., 1])
    return result
