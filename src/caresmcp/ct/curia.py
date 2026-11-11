"""Curia's supported two-dimensional CT classification heads."""

from typing import Any

import numpy as np

from caresmcp.api.decorators import requires
from caresmcp.api.io import get_classification_scores
from caresmcp.api.models.hf_hub import (
    HFModelType,
    get_hf_json,
    get_hf_model_and_processor,
)
from caresmcp.constants import DEVICE
from caresmcp.registry import CT, TASK_CLS

from ._arg import (
    ClassifyCtSliceWithCuriaAnatomicalPlaneArg,
    ClassifyCtSliceWithCuriaClassificationTaskArg,
    CTImage,
    CuriaCTPlane,
    RasterRescaleIntercept,
    RasterRescaleSlope,
)
from .io import DecodedCTSlice, _decode_ct_slice, _is_dicom

CURIA_MODEL_ID = "raidium/curia"


def _curia_hu_input(decoded: DecodedCTSlice, plane: CuriaCTPlane) -> np.ndarray:
    """Orient orthogonal DICOM pixel axes to the checkpoint's row/column axes."""
    hu = decoded.require_hu()
    if decoded.image_orientation_patient is None:
        return hu
    row, column = {
        "axial": ((0, 1, 0), (1, 0, 0)),  # PL
        "coronal": ((0, 0, -1), (1, 0, 0)),  # IL
        "sagittal": ((0, 0, -1), (0, 1, 0)),  # IP
    }[plane]
    orientation = np.asarray(decoded.image_orientation_patient).reshape(2, 3)
    # DICOM's first direction vector describes increasing column index.
    axes = orientation[::-1]
    if np.allclose(np.abs(axes), np.abs((column, row)), atol=1e-3):
        hu = hu.T
        axes = axes[::-1]
    if not np.allclose(np.abs(axes), np.abs((row, column)), atol=1e-3):
        raise ValueError(
            "Curia requires orthogonal DICOM axes matching the selected plane"
        )
    if np.dot(axes[0], row) < 0:
        hu = hu[::-1]
    if np.dot(axes[1], column) < 0:
        hu = hu[:, ::-1]
    return np.ascontiguousarray(hu)


@CT.register("classify_ct_slice_with_curia")
@TASK_CLS.register("classify_ct_slice_with_curia")
@requires(["torch", "transformers", "huggingface_hub"], extra="hf")
def classify_ct_slice_with_curia(
    image_or_dicom: CTImage,
    classification_task: ClassifyCtSliceWithCuriaClassificationTaskArg = "ct_anatomy_classification",
    anatomical_plane: ClassifyCtSliceWithCuriaAnatomicalPlaneArg = "axial",
    raster_hu_rescale_slope: RasterRescaleSlope = 1.0,
    raster_hu_rescale_intercept: RasterRescaleIntercept = 0.0,
) -> dict[str, Any]:
    """Classify CT anatomy, COVID/pneumonia, or hemorrhage with a Curia slice head.

    Input must contain calibrated, unwindowed HU. Rasters must already have
    row/column directions PL for axial, IL for coronal, or IP for sagittal;
    orthogonal DICOM axes are reoriented using their orientation metadata.
    COVID and hemorrhage heads accept axial slices only. Scores concern one
    slice, not an entire examination or a calibrated clinical probability.

    Curia requires approved Hugging Face access and uses the Research-only
    RAIL-M license. Dancette et al., "Curia: A Multi-Modal Foundation Model
    for Radiology" (2025), https://arxiv.org/abs/2509.06830.

    Returns:
        Model and head provenance, input plane, and labeled softmax scores.

    See Also:
        https://huggingface.co/raidium/curia
    """
    import torch

    head = {
        "ct_anatomy_classification": "anatomy-ct",
        "covid_pneumonia_classification": "covidx-ct",
        "intracranial_hemorrhage_classification": "ich",
    }[classification_task]
    if head != "anatomy-ct" and anatomical_plane != "axial":
        raise ValueError("The selected Curia head requires axial CT slices")
    decoded = _decode_ct_slice(
        image_or_dicom,
        raster_hu_rescale_slope,
        raster_hu_rescale_intercept,
        "hu",
    )
    if _is_dicom(image_or_dicom) and decoded.image_orientation_patient is None:
        raise ValueError("Curia CT DICOM input requires image orientation metadata")
    hu = _curia_hu_input(decoded, anatomical_plane)
    model, processor = get_hf_model_and_processor(
        CURIA_MODEL_ID,
        HFModelType.IMAGE_CLASSIFICATION,
        trust_remote_code=True,
        use_fast=False,
        subfolder=head,
        processor_subfolder="",
    )
    inputs = processor(images=hu, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs)["logits"]
    labels = get_hf_json(CURIA_MODEL_ID, "id_to_labels.json").get(head)
    if not isinstance(labels, dict):
        raise TypeError(
            "Curia checkpoint does not provide labels for the selected head"
        )
    return {
        "model_id": CURIA_MODEL_ID,
        "head": head,
        "task": classification_task,
        "prediction_level": "slice",
        "plane": anatomical_plane,
        "score_mode": "softmax",
        "intensity_units": "hu",
        "scores": get_classification_scores(logits, labels, "softmax"),
    }
