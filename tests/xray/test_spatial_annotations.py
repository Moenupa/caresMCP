"""Measure retained spatial tools against official manual lung annotations.

Source: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/
Montgomery cases are distinct patients, one normal and one abnormal. Training
cohort overlap is unknown; these small checks do not establish held-out accuracy.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import encode_as
from caresmcp.xray.biovil import biovil_ground_phrases
from caresmcp.xray.flexray import flexray_segment_anatomy
from caresmcp.xray.torchxrayvision import torchxrayvision_segment_anatomy

pytestmark = pytest.mark.slow

MONTGOMERY_ROOT = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Montgomery-County-CXR-Set/MontgomerySet/"
)


@pytest.fixture(scope="module", params=["MCUCXR_0001_0", "MCUCXR_0104_1"])
def manual_lung_example(request):
    """Download original radiographs and separate official manual lung masks."""
    filename = request.param + ".png"
    image = _read_bytes(MONTGOMERY_ROOT + "CXR_png/" + filename)
    masks = {}
    for side in ("left", "right"):
        content = _read_bytes(MONTGOMERY_ROOT + f"ManualMask/{side}Mask/" + filename)
        with Image.open(BytesIO(content)) as mask:
            masks[side] = np.asarray(mask.convert("L")) > 0
    with Image.open(BytesIO(image)) as source:
        width, height = source.size
    assert all(mask.shape == (height, width) for mask in masks.values())
    assert all(mask.any() for mask in masks.values())
    return image, masks, filename


def _overlap_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Report overlap without deriving reference labels from image intensities."""
    assert prediction.shape == target.shape
    intersection = int(np.count_nonzero(prediction & target))
    predicted = int(np.count_nonzero(prediction))
    reference = int(np.count_nonzero(target))
    return {
        "dice": 2 * intersection / (predicted + reference),
        "iou": intersection / (predicted + reference - intersection),
        "predicted_pixels": predicted,
        "reference_pixels": reference,
    }


@pytest.mark.parametrize(
    "model_id", [pytest.param("flexray", marks=pytest.mark.cuda), "torchxrayvision"]
)
@save_returns("xray_manual_masks", unpack=True)
def test_anatomy_manual_lung_masks(manual_lung_example, model_id):
    """Measure lung Dice/IoU in each model's documented output geometry."""
    image, masks, filename = manual_lung_example
    target = masks["left"] | masks["right"]
    if model_id == "flexray":
        pytest.importorskip("fxr")
        result = flexray_segment_anatomy(image, region="lungs")
        geometry = "original image coordinates"
    else:
        pytest.importorskip("torchxrayvision")
        left = torchxrayvision_segment_anatomy(image, region="Left Lung")
        right = torchxrayvision_segment_anatomy(image, region="Right Lung")
        with Image.open(BytesIO(left)) as lm, Image.open(BytesIO(right)) as rm:
            merged = (np.asarray(lm) > 0) | (np.asarray(rm) > 0)
        result = encode_as(Image.fromarray(merged.astype(np.uint8) * 255))
        height, width = target.shape
        side = min(height, width)
        # XRayCenterCrop uses integer-truncated offsets, then a 512-square resize.
        top, left = height // 2 - side // 2, width // 2 - side // 2
        target = target[top : top + side, left : left + side]
        target = np.asarray(
            Image.fromarray(target).resize((512, 512), Image.Resampling.NEAREST)
        )
        geometry = "512-square center crop"
    with Image.open(BytesIO(result)) as mask:
        prediction = np.asarray(mask) > 0
    assert prediction.any(), (
        "Expected nonempty lung predictions on annotated radiographs"
    )
    metrics = _overlap_metrics(prediction, target)
    if model_id == "torchxrayvision":
        # Both reference cases have measured Dice above 0.87 with this model.
        assert metrics["dice"] > 0.8
    return {
        "prediction": result,
        "manual_reference": encode_as(Image.fromarray(target.astype(np.uint8) * 255)),
        "metrics": {
            "source_image": MONTGOMERY_ROOT + "CXR_png/" + filename,
            "source_manual_masks": MONTGOMERY_ROOT + "ManualMask/",
            "model": model_id,
            "geometry": geometry,
            "training_overlap": "unknown",
            **metrics,
        },
    }


@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_id",
    ["microsoft/BiomedVLP-CXR-BERT-specialized", "microsoft/BiomedVLP-BioViL-T"],
)
@save_returns("xray_manual_grounding", unpack=True)
def test_biovil_manual_lung_regions(manual_lung_example, model_id):
    """Check phrase-map peaks and foreground/background ranking on manual masks."""
    pytest.importorskip("health_multimodal")
    image, masks, filename = manual_lung_example
    result = biovil_ground_phrases(
        image, ["left lung", "right lung"], model_id=model_id
    )
    left, top, right, bottom = result["coverage_xyxy_original_pixels"]
    metrics = []
    for side, phrase in zip(("left", "right"), result["phrases"], strict=True):
        scores = np.asarray(phrase["patch_similarity_map"])
        assert np.isfinite(scores).all()
        rows, columns = scores.shape
        xs = (left + (np.arange(columns) + 0.5) * (right - left) / columns).astype(int)
        ys = (top + (np.arange(rows) + 0.5) * (bottom - top) / rows).astype(int)
        # Official folders name image sides: leftMask is image-left. In these
        # standard PA projections the patient's left lung is image-right.
        image_side = "right" if side == "left" else "left"
        target = masks[image_side][ys[:, None], xs[None, :]]
        assert target.any() and not target.all()
        positive, negative = scores[target], scores[~target]
        # Pairwise AUC accounts for ties without imposing a map threshold.
        auc = float(
            (
                (positive[:, None] > negative) + 0.5 * (positive[:, None] == negative)
            ).mean()
        )
        peak = np.unravel_index(scores.argmax(), scores.shape)
        metrics.append(
            {
                "phrase": phrase["phrase"],
                "peak_inside_manual_region": bool(target[peak]),
                "grid_pixel_auc": auc,
                "manual_positive_grid_cells": int(target.sum()),
            }
        )
    return {
        "grounding": result,
        "metrics": {
            "source_image": MONTGOMERY_ROOT + "CXR_png/" + filename,
            "source_manual_masks": MONTGOMERY_ROOT + "ManualMask/",
            "training_overlap": "unknown",
            "laterality": "PA patient-left = image-right; NLM folder names use image sides",
            "regions": metrics,
        },
    }
