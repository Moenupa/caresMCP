"""Adjacent-slice noncontrast head CT hemorrhage classification."""

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from caresmcp.api.decorators import requires
from caresmcp.api.io import device_copy, get_classification_scores
from caresmcp.api.models.hf_hub import get_hf_checkpoint
from caresmcp.api.models.timm import build_timm_model
from caresmcp.constants import DEVICE
from caresmcp.registry import CT, TASK_CLS

from ._arg import (
    ClassifyCtHeadHemorrhageNextImageArg,
    ClassifyCtHeadHemorrhagePreviousImageArg,
    CTImage,
    HURasterUnits,
    RasterRescaleIntercept,
    RasterRescaleSlope,
)
from .io import _decode_ct_slice

if TYPE_CHECKING:
    from torch.nn import Module


HEMORRHAGE_MODEL_ID = "ianpan/ct-head-hemorrhage-detection"

HEMORRHAGE_LABELS = {
    0: "epidural",
    1: "intraparenchymal",
    2: "intraventricular",
    3: "subarachnoid",
    4: "subdural",
    5: "any",
}


def get_ct_hemorrhage_classifier() -> "Module":
    """Build the baseline slice classifier, excluding sequence and mask heads."""
    from timm.layers import SelectAdaptivePool2d
    from torch import nn

    # Match skp.models.classification.net2d.Net at source revision 914c57d.
    # Normalization is performed on the windowed slab by the CT adapter.
    backbone = build_timm_model(
        "maxvit_tiny_tf_512.in1k",
        pretrained=False,
        num_classes=0,
        global_pool="",
        features_only=False,
        in_chans=9,
    )
    model = nn.Sequential(
        OrderedDict(
            backbone=backbone,
            pooling=SelectAdaptivePool2d(pool_type="avg", flatten=True),
            dropout=nn.Dropout(0.2),
            linear=nn.Linear(512, 6),
        )
    )
    state = get_hf_checkpoint(HEMORRHAGE_MODEL_ID, "classifier/model.safetensors")
    model.load_state_dict(state, strict=True)
    return device_copy(model)


def _window_hemorrhage_slice(hu: Any | None) -> np.ndarray:
    """Match upstream window_hu_slice, including padding after windowing."""
    if hu is None:
        return np.zeros((3, 512, 512), dtype=np.float32)
    if hu.shape[0] > 640 or hu.shape[1] > 640:
        hu = cv2.resize(hu, (512, 512), interpolation=cv2.INTER_AREA)
    top = max((hu.shape[0] - 512) // 2, 0)
    left = max((hu.shape[1] - 512) // 2, 0)
    hu = hu[top : top + 512, left : left + 512]
    centers = np.array([40, 80, 600], dtype=np.float32)[:, None, None]
    widths = np.array([80, 200, 2800], dtype=np.float32)[:, None, None]
    lower = centers - widths / 2
    windowed = (np.clip(hu[None], lower, centers + widths / 2) - lower) / (
        widths + 1e-6
    )
    height = 512 - hu.shape[0]
    width = 512 - hu.shape[1]
    return np.pad(
        windowed,
        ((0, 0), (height // 2, height - height // 2), (width // 2, width - width // 2)),
    )


@CT.register("classify_ct_head_hemorrhage")
@TASK_CLS.register("classify_ct_head_hemorrhage")
@requires(["timm"], extra="timm")
def classify_ct_head_hemorrhage(
    image_or_dicom: CTImage,
    previous_image_or_dicom: ClassifyCtHeadHemorrhagePreviousImageArg = b"",
    next_image_or_dicom: ClassifyCtHeadHemorrhageNextImageArg = b"",
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    raster_units: HURasterUnits = "display",
) -> dict[str, Any]:
    """Score six overlapping hemorrhage labels for one noncontrast axial head CT slice.

    Supply the immediately previous and next slices from the same physically
    ordered series in inferior-to-superior order. Omit a neighbor only at a
    series boundary. Inputs must contain calibrated HU in the same axial
    orientation and pixel grid.
    Raster inputs must declare their HU calibration; display JPEGs are rejected.

    Scores describe the baseline slice classifier for the center slice. They
    are independent sigmoid scores, not a series-level diagnosis or the
    contextual sequence ensemble. This research model is not a medical device.

    Returns:
        Checkpoint provenance, neighbor availability, and six labeled scores.

    See Also:
        https://huggingface.co/ianpan/ct-head-hemorrhage-detection
    """
    import torch

    decoded = [
        _decode_ct_slice(source, rescale_slope, rescale_intercept, raster_units)
        if source is not None
        else None
        for source in (
            previous_image_or_dicom or None,
            image_or_dicom,
            next_image_or_dicom or None,
        )
    ]
    center = decoded[1]
    if center is None:
        raise ValueError("image_or_dicom must contain the center CT slice")
    for item in decoded:
        if item is None:
            continue
        item.require_hu()
        if item.values.shape != center.values.shape:
            raise ValueError("Adjacent CT slices must have matching dimensions")
        if item.image_orientation_patient is not None and not np.allclose(
            item.image_orientation_patient, (1, 0, 0, 0, 1, 0), atol=1e-3
        ):
            raise ValueError("Head CT DICOM slices must have axial LPS orientation")
        if (
            item.pixel_spacing_row_column_mm is not None
            and center.pixel_spacing_row_column_mm is not None
            and not np.allclose(
                item.pixel_spacing_row_column_mm, center.pixel_spacing_row_column_mm
            )
        ):
            raise ValueError("Adjacent CT slices must have matching pixel spacing")
    # Validate DICOM geometry when available. Raster ordering is caller supplied.
    positions = [item.image_position_patient if item else None for item in decoded]
    if positions[1] is not None:
        for index in (0, 2):
            position = positions[index]
            if position is not None and not np.allclose(
                position[:2], positions[1][:2], atol=1e-3
            ):
                raise ValueError("Adjacent CT slices must share the same pixel grid")
        if positions[0] is not None and positions[0][2] >= positions[1][2]:
            raise ValueError(
                "previous_image_or_dicom must precede image_or_dicom in inferior-to-superior order"
            )
        if positions[2] is not None and positions[2][2] <= positions[1][2]:
            raise ValueError(
                "next_image_or_dicom must follow image_or_dicom in inferior-to-superior order"
            )
    slab = np.concatenate(
        [_window_hemorrhage_slice(item.values if item else None) for item in decoded]
    )
    inputs = torch.from_numpy(slab).unsqueeze(0).to(DEVICE) * 2 - 1
    model = get_ct_hemorrhage_classifier()
    with torch.inference_mode():
        logits = model(inputs)
    return {
        "model_id": HEMORRHAGE_MODEL_ID,
        "task": "intracranial_hemorrhage_classification",
        "prediction_level": "center_slice",
        "classifier": "baseline_slice",
        "score_mode": "sigmoid",
        "intensity_units": "hu",
        "previous_slice_present": bool(previous_image_or_dicom),
        "next_slice_present": bool(next_image_or_dicom),
        "scores": get_classification_scores(logits, HEMORRHAGE_LABELS, "sigmoid"),
    }
