"""DALE-CT-2S chest CT feature extraction.

The checkpoint weights carry a CC-BY-NC-SA-4.0 license and are intended for
non-commercial research use. This tool does not provide a diagnosis.
"""

from typing import Any

from caresmcp.api.decorators import requires
from caresmcp.api.models import get_timm_model
from caresmcp.constants import DEVICE
from caresmcp.registry import CT, TASK_EMB

from ._arg import (
    CTImage,
    HURasterUnits,
    RasterRescaleIntercept,
    RasterRescaleSlope,
)
from .io import _decode_ct_slice

# DALE-CT-2S is a fixed encoder adapter. Keep model provenance, architecture,
# artifact name, and preprocessing statistics together rather than exposing
# them as caller-selectable model paths or revisions.
CT_DALE_MODEL_ID = "hf-hub:Kentucky-Open-Science/DALE-CT-2S"
CT_DALE_PATCH_SIZE = 16
CT_DALE_MIN_HU = -997.0
CT_DALE_MAX_HU = 888.0
CT_DALE_MEAN_HU = -142.39
CT_DALE_STD_HU = 360.97


def _preprocess_dale_ct_2s(hu: Any) -> Any:
    """Apply the checkpoint's HU clipping, scaling, and training normalization."""
    import torch

    if getattr(hu, "ndim", None) != 2:
        raise ValueError("CT slice must be a two-dimensional array")
    height, width = hu.shape
    if height % CT_DALE_PATCH_SIZE or width % CT_DALE_PATCH_SIZE:
        raise ValueError(
            f"CT slice height and width must be divisible by {CT_DALE_PATCH_SIZE}"
        )

    values = torch.as_tensor(hu, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("CT slice must contain only finite pixel values")
    value_range = CT_DALE_MAX_HU - CT_DALE_MIN_HU
    normalized_mean = (CT_DALE_MEAN_HU - CT_DALE_MIN_HU) / value_range
    normalized_std = CT_DALE_STD_HU / value_range
    values = torch.clamp(values, CT_DALE_MIN_HU, CT_DALE_MAX_HU)
    values = (values - CT_DALE_MIN_HU) / value_range
    values = (values - normalized_mean) / normalized_std
    return values.unsqueeze(0).unsqueeze(0).to(DEVICE)


@CT.register("dale_ct_2s_image_embed")
@TASK_EMB.register("dale_ct_2s_image_embed")
@requires(["torch", "timm", "transformers"], extra="ct")
def dale_ct_2s_image_embed(
    image_or_dicom: CTImage,
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    raster_units: HURasterUnits = "display",
) -> dict[str, Any]:
    """Extract a global DALE-CT-2S feature vector from one chest CT slice.

    Requires calibrated, unwindowed HU and slice dimensions divisible by 16.

    Returns:
        Model provenance, input dimensions, and a JSON-safe embedding.

    See Also:
        https://huggingface.co/Kentucky-Open-Science/DALE-CT-2S
        https://github.com/Kentucky-Open-Science/DALE-CT
    """
    import torch

    decoded = _decode_ct_slice(
        image_or_dicom, rescale_slope, rescale_intercept, raster_units
    )
    model_input = _preprocess_dale_ct_2s(decoded.require_hu())
    model = get_timm_model(CT_DALE_MODEL_ID, pretrained=True)
    with torch.inference_mode():
        embedding = model(model_input)
    values = embedding[0].detach().float().cpu().tolist()
    return {
        "model_id": CT_DALE_MODEL_ID,
        "embedding_type": "global",
        "image_shape_hw": [int(decoded.values.shape[0]), int(decoded.values.shape[1])],
        "embedding": [float(value) for value in values],
    }
