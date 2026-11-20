"""Reproduce the Stenoz coronary U-Nets on the release's official samples.

Every input is one of the ten ARCADE frames and cardiologist stenosis boxes
shipped with https://github.com/uzbtrust/stenoz at a pinned revision. The
release reports held-out ARCADE scores of F1 0.645, lesion recall 95.7%, and
image-level sensitivity 95.7% for the stenosis detector, and Dice 0.702 for
vessel segmentation (``models/metrics_stenosis.json`` and
``models/metrics.json``). Those dataset-level scores need the ARCADE splits,
which the release does not ship, so they cannot be re-measured here. Instead
these tests pin the published validation scores carried by the released
checkpoints, reproduce the release's own inference algorithm on the official
frames, and require every frame to localize inside its released cardiologist
boxes.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as
from caresmcp.api.metrics import mask_overlap
from caresmcp.api.models import get_hf_checkpoint
from caresmcp.dsa import coronary

if TYPE_CHECKING:
    from torch.nn import Module

pytestmark = pytest.mark.slow

# Published validation scores from the release's model metric files; both
# released checkpoints carry these exact values next to their state
# dictionaries.
PUBLISHED_VAL_F1 = 0.7362579659641502
PUBLISHED_VAL_DICE = 0.7682008481025696
PUBLISHED_SIZE = 512
PUBLISHED_BASE = 32

# Released evaluation protocol: a detection is localized when its center lies
# inside a cardiologist box with 6 px tolerance (``src/eval_stenosis_dl.py``
# and the paper's local validation criterion).
LOCALIZATION_PAD_PX = 6.0
STENOSIS_AREA_PX = 40
VESSEL_AREA_PX = 60

# Two independent CUDA runs may accumulate convolutions in a different order,
# so compare boundary-sensitive quantities with tight tolerances instead of bit
# equality. The same comparison is exact on CPU.
CENTROID_TOLERANCE_PX = 0.05
SCORE_TOLERANCE = 1e-3
AREA_TOLERANCE_PX = 4
MINIMUM_REFERENCE_IOU = 0.99


def _reference_probabilities(image: bytes, model: "Module") -> np.ndarray:
    """Apply the release's min-max scaling, reflect padding, and sigmoid.

    Transcribed from ``src/dlstenosis.py`` and ``src/dlseg.py`` in the pinned
    Stenoz revision so the tools can be compared with the release's algorithm.
    """
    import torch

    pixels = np.asarray(decode_as(image).convert("L"), dtype=np.float32)
    pixels = (pixels - pixels.min()) / (np.ptp(pixels) + 1e-6)
    height, width = pixels.shape
    padded = np.pad(pixels, ((0, (-height) % 16), (0, (-width) % 16)), mode="reflect")
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(torch.from_numpy(padded)[None, None].to(device))
    return logits[0, 0].sigmoid().float().cpu().numpy()[:height, :width]


def _reference_detections(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_area: int = STENOSIS_AREA_PX,
) -> list[tuple[float, float, float, int]]:
    """Return the release's center, peak score, and area per component."""
    from scipy import ndimage

    labels, count = ndimage.label(probabilities > threshold)
    detections = []
    for label in range(1, count + 1):
        rows, columns = np.nonzero(labels == label)
        if rows.size < min_area:
            continue
        detections.append(
            (
                float(columns.mean()),
                float(rows.mean()),
                float(probabilities[rows, columns].max()),
                int(rows.size),
            )
        )
    detections.sort(key=lambda detection: -detection[2])
    return detections


def _reference_mask(
    probabilities: np.ndarray, threshold: float, min_area: int
) -> np.ndarray:
    """Keep connected components at or above the released minimum area."""
    from scipy import ndimage

    labels, _ = ndimage.label(probabilities > threshold)
    keep = np.bincount(labels.ravel()) >= min_area
    keep[0] = False
    return keep[labels]


def _inside_any_box(
    x: float, y: float, boxes: list[list[float]], pad: float = 0.0
) -> bool:
    """Return whether a point falls inside a released cardiologist box."""
    return any(
        left - pad <= x <= left + width + pad and top - pad <= y <= top + height + pad
        for left, top, width, height in boxes
    )


@skip_on_import_error
def test_coronary_checkpoints_reproduce_published_scores():
    """Check the released weights carry the published validation scores.

    The ARCADE test and validation splits are not part of the release, so the
    held-out scores cannot be re-measured from the official repository. The
    released checkpoints do store the validation scores written by the same
    training run that produced the metric files, so pinning those values ties
    the tools to the exact published weights.
    """
    stenosis = get_hf_checkpoint(coronary.CORONARY_MODEL_ID, "unet_stenosis.pt")
    vessel = get_hf_checkpoint(coronary.CORONARY_MODEL_ID, "unet_vessel.pt")

    assert stenosis["base"] == vessel["base"] == PUBLISHED_BASE
    assert stenosis["size"] == vessel["size"] == PUBLISHED_SIZE
    assert stenosis["val_f1"] == PUBLISHED_VAL_F1
    assert vessel["val_dice"] == PUBLISHED_VAL_DICE


@skip_on_import_error
@save_returns("dsa", unpack=True)
def test_detect_coronary_stenosis(coronary_samples, coronary_annotations):
    """Check every official frame localizes and matches the released algorithm."""
    model = coronary._load_model("stenosis")
    artifacts: dict[str, Any] = {}
    for name, image in coronary_samples.items():
        result = coronary.detect_coronary_stenosis(image)

        assert result["model_id"] == coronary.CORONARY_MODEL_ID
        assert result["checkpoint"] == "unet_stenosis.pt"
        assert result["task"] == "coronary_stenosis_detection"
        assert result["score_mode"] == "sigmoid"
        assert result["image_width"] == result["image_height"] == PUBLISHED_SIZE
        assert result["score_threshold"] == 0.5
        assert result["min_area_pixels"] == STENOSIS_AREA_PX
        assert result["polarity_inverted"] is False
        assert result["detection_count"] == len(result["detections"])
        assert result["detection_count"] > 0
        assert result["max_detection_score"] == max(
            detection["peak_score"] for detection in result["detections"]
        )
        for previous, current in zip(result["detections"], result["detections"][1:]):
            assert previous["peak_score"] >= current["peak_score"]
        for detection in result["detections"]:
            x, y = detection["centroid_xy"]
            assert 0 <= x < result["image_width"]
            assert 0 <= y < result["image_height"]
            assert 0 < detection["peak_score"] <= 1
            assert detection["area_pixels"] >= result["min_area_pixels"]

        # The release's image-level sensitivity criterion on official frames:
        # at least one detection center lies inside a cardiologist box.
        boxes = coronary_annotations[f"{name}.png"]["boxes"]
        assert any(
            _inside_any_box(x, y, boxes, LOCALIZATION_PAD_PX)
            for x, y in (detection["centroid_xy"] for detection in result["detections"])
        ), f"{name}.png: no detection localized inside a released box"

        reference = _reference_detections(_reference_probabilities(image, model))
        measured = [
            (
                detection["centroid_xy"][0],
                detection["centroid_xy"][1],
                detection["peak_score"],
                detection["area_pixels"],
            )
            for detection in result["detections"]
        ]
        assert len(measured) == len(reference), (
            f"{name}.png: {len(measured)} detections but the release "
            f"algorithm found {len(reference)}"
        )
        for expected, actual in zip(reference, measured):
            assert actual[0] == pytest.approx(expected[0], abs=CENTROID_TOLERANCE_PX), (
                f"{name}.png: centroid x differs from the release algorithm"
            )
            assert actual[1] == pytest.approx(expected[1], abs=CENTROID_TOLERANCE_PX), (
                f"{name}.png: centroid y differs from the release algorithm"
            )
            assert actual[2] == pytest.approx(expected[2], abs=SCORE_TOLERANCE), (
                f"{name}.png: peak score differs from the release algorithm"
            )
            assert abs(actual[3] - expected[3]) <= AREA_TOLERANCE_PX, (
                f"{name}.png: component area differs from the release algorithm"
            )
        artifacts[f"{name}_detections"] = result
    return artifacts


@skip_on_import_error
@save_returns("dsa", unpack=True)
def test_segment_coronary_stenosis(coronary_samples):
    """Check the candidate-lesion mask matches the released segmentation rule."""
    image = coronary_samples["126"]
    result = coronary.segment_coronary_stenosis(image)

    mask = decode_as(result)
    labels = np.asarray(mask)
    assert mask.size == (PUBLISHED_SIZE, PUBLISHED_SIZE)
    assert mask.mode == "P"
    assert mask.info.get("transparency") == 0
    assert labels.max() == 1
    assert (labels == 1).sum() > 0

    model = coronary._load_model("stenosis")
    reference = _reference_mask(
        _reference_probabilities(image, model), 0.5, STENOSIS_AREA_PX
    )
    overlap = mask_overlap(labels == 1, reference)
    assert overlap.iou is not None, "reference stenosis mask must not be empty"
    assert overlap.iou >= MINIMUM_REFERENCE_IOU, (
        f"mask IoU {overlap.iou:.4f} does not reproduce the release algorithm"
    )
    return {"mask": result}


@skip_on_import_error
@save_returns("dsa", unpack=True)
def test_segment_coronary_vessels(coronary_samples):
    """Check the vessel mask matches the released segmentation rule."""
    image = coronary_samples["126"]
    result = coronary.segment_coronary_vessels(image)

    mask = decode_as(result)
    labels = np.asarray(mask)
    assert mask.size == (PUBLISHED_SIZE, PUBLISHED_SIZE)
    assert mask.mode == "P"
    assert mask.info.get("transparency") == 0
    assert labels.max() == 1
    assert (labels == 1).sum() > 0

    model = coronary._load_model("vessel")
    reference = _reference_mask(
        _reference_probabilities(image, model), 0.5, VESSEL_AREA_PX
    )
    overlap = mask_overlap(labels == 1, reference)
    assert overlap.iou is not None, "reference vessel mask must not be empty"
    assert overlap.iou >= MINIMUM_REFERENCE_IOU, (
        f"mask IoU {overlap.iou:.4f} does not reproduce the release algorithm"
    )
    return {"mask": result}
