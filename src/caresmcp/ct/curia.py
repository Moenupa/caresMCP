"""Curia's supported two-dimensional CT classification heads."""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, validate_call

from caresmcp.api.decorators import requires
from caresmcp.api.models.hf_hub import (
    HFModelType,
    get_hf_json,
    get_hf_model_and_processor,
)
from caresmcp.api.tensorio import get_classification_scores
from caresmcp.constants import DEVICE
from caresmcp.registry import CT, TASK_CLS

from .io import DecodedCTSlice, _decode_ct_slice

CuriaCTTask = Literal[
    "ct_anatomy_classification",
    "covid_pneumonia_classification",
    "intracranial_hemorrhage_classification",
]
CuriaCTPlane = Literal["axial", "coronal", "sagittal"]


DICOM_OR_RASTER = Field(
    min_length=1,
    description="Single CT DICOM slice or calibrated grayscale raster bytes.",
)
HUGGINGFACE_MODEL_ID = Field(description="Curia checkpoint repository.")
CLASSIFICATION_TASK = Field(
    description="Anatomy, COVID/pneumonia, or hemorrhage classification head."
)
ANATOMICAL_PLANE = Field(description="Anatomical plane of the supplied CT slice.")
CT_INPUT_FORMAT = Field(
    description="Auto-detect the format, or explicitly select DICOM or a grayscale raster."
)
RASTER_HU_RESCALE_SLOPE = Field(
    allow_inf_nan=False,
    description="Raster HU = stored value * slope + intercept; ignored for DICOM.",
)
RASTER_HU_RESCALE_INTERCEPT = Field(
    allow_inf_nan=False,
    description="Offset in HU after multiplying raster values by the slope; ignored for DICOM.",
)
CT_RASTER_UNITS = Field(
    description="Raster values must have known Hounsfield-unit calibration; ignored for DICOM."
)


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
@validate_call
@requires(["torch", "transformers", "huggingface_hub"], extra="hf")
def classify_ct_slice_with_curia(
    dicom_or_raster: Annotated[bytes, DICOM_OR_RASTER],
    huggingface_model_id: Annotated[
        Literal["raidium/curia"], HUGGINGFACE_MODEL_ID
    ] = "raidium/curia",
    classification_task: Annotated[
        CuriaCTTask, CLASSIFICATION_TASK
    ] = "ct_anatomy_classification",
    anatomical_plane: Annotated[CuriaCTPlane, ANATOMICAL_PLANE] = "axial",
    input_format: Annotated[
        Literal["auto", "dicom", "grayscale_raster"], CT_INPUT_FORMAT
    ] = "auto",
    raster_hu_rescale_slope: Annotated[float, RASTER_HU_RESCALE_SLOPE] = 1.0,
    raster_hu_rescale_intercept: Annotated[float, RASTER_HU_RESCALE_INTERCEPT] = 0.0,
    raster_units: Annotated[Literal["hu"], CT_RASTER_UNITS] = "hu",
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
        dicom_or_raster,
        "image" if input_format == "grayscale_raster" else input_format,
        raster_hu_rescale_slope,
        raster_hu_rescale_intercept,
        raster_units,
    )
    is_dicom = input_format == "dicom" or (
        input_format == "auto"
        and len(dicom_or_raster) >= 132
        and dicom_or_raster[128:132] == b"DICM"
    )
    if is_dicom and decoded.image_orientation_patient is None:
        raise ValueError("Curia CT DICOM input requires image orientation metadata")
    hu = _curia_hu_input(decoded, anatomical_plane)
    model, processor = get_hf_model_and_processor(
        huggingface_model_id,
        HFModelType.IMAGE_CLASSIFICATION,
        trust_remote_code=True,
        use_fast=False,
        subfolder=head,
        processor_subfolder="",
    )
    inputs = processor(images=hu, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs)["logits"]
    labels = get_hf_json(huggingface_model_id, "id_to_labels.json").get(head)
    if not isinstance(labels, dict):
        raise TypeError(
            "Curia checkpoint does not provide labels for the selected head"
        )
    return {
        "model_id": huggingface_model_id,
        "head": head,
        "task": classification_task,
        "prediction_level": "slice",
        "plane": anatomical_plane,
        "score_mode": "softmax",
        "intensity_units": "hu",
        "scores": get_classification_scores(logits, labels, "softmax"),
    }
