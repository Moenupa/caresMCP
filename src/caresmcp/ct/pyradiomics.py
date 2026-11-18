"""PyRadiomics-compatible CT slice feature extraction.

Uses the pure-Python ``numpyradiomics`` backend, which runs on CPU, and
normalizes its feature names and default feature set to PyRadiomics
conventions.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_mask
from caresmcp.registry import CT

from ._arg import (
    CTImage,
    ExtractCtSliceRadiomicsPixelSpacingRowColumnMmArg,
    HURasterUnits,
    Mask,
    PositiveFloat,
    RasterRescaleIntercept,
    RasterRescaleSlope,
)
from .io import _decode_ct_slice

_GLRLM_RENAMES = {
    "LongRunHighGrayLevelRunEmphasis": "LongRunHighGrayLevelEmphasis",
    "LongRunLowGrayLevelRunEmphasis": "LongRunLowGrayLevelEmphasis",
    "ShortRunHighGrayLevelRunEmphasis": "ShortRunHighGrayLevelEmphasis",
    "ShortRunLowGrayLevelRunEmphasis": "ShortRunLowGrayLevelEmphasis",
    "RunLengthVariance": "RunVariance",
}
_GLSZM_RENAMES = {
    "ZoneSizeNonUniformity": "SizeZoneNonUniformity",
    "ZoneSizeNonUniformityNormalized": "SizeZoneNonUniformityNormalized",
    "ZoneSizeVariance": "ZoneVariance",
}


def _prefixed(
    prefix: str,
    values: dict[str, float],
    renames: dict[str, str] | None = None,
) -> dict[str, float]:
    renames = renames or {}
    return {
        f"original_{prefix}_{renames.get(name, name)}": float(value)
        for name, value in values.items()
    }


def _glrlm_2d(
    image: NDArray[np.float32],
    mask: NDArray[np.bool_],
    bin_width: float,
) -> dict[str, float]:
    """Compute 2D GLRLM features with PyRadiomics feature names.

    numpyradiomics 0.0.5 reads a third direction component for 2D input, so
    this mirrors its GLRLM loop with the four planar directions padded by a
    zero third component.
    """
    from numpyradiomics import mod_glrlm

    quantized = mod_glrlm._discretize_image(
        image, mask, binWidth=bin_width, binCount=None
    )
    levels = int(quantized.max())
    max_length = max(quantized.shape)
    npixels = int(np.count_nonzero(mask))
    totals: dict[str, float] = {}
    angles = 0
    for row_step, column_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        runs, lengths = mod_glrlm._compute_runs_skewed(
            quantized, (row_step, column_step, 0)
        )
        if runs.size == 0:
            continue
        matrix = np.zeros((levels, max_length + 1), dtype=np.float64)
        np.add.at(
            matrix,
            (runs - 1, np.clip(lengths, 1, max_length + 1) - 1),
            1,
        )
        values = mod_glrlm._compute_glrlm_features(matrix, npixels)
        if not values:
            continue
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        angles += 1
    if not angles:
        return {}
    return {name: value / angles for name, value in totals.items()}


@CT.register("extract_ct_slice_radiomics")
@requires(["numpyradiomics"], extra="ct")
def radiomics_ct_slice(
    image_or_dicom: CTImage,
    roi_mask: Mask,
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    pixel_spacing_row_column_mm: ExtractCtSliceRadiomicsPixelSpacingRowColumnMmArg = None,
    bin_width: PositiveFloat = 25.0,
    raster_units: HURasterUnits = "display",
) -> dict[str, Any]:
    """Extract two-dimensional PyRadiomics features from one CT slice ROI.

    Feature values depend on the ROI, HU conversion, pixel spacing, bin width,
    and radiomics implementation version. The supplied mask defines the ROI;
    this function does not segment or classify it.

    Returns:
        Finite two-dimensional radiomics features, HU intensity units, and
        the spacing source ('provided', 'dicom', or 'assumed'). Diagnostic
        entries and non-finite feature values are omitted.
    """
    import numpyradiomics

    decoded = _decode_ct_slice(
        image_or_dicom, rescale_slope, rescale_intercept, raster_units
    )
    hu = decoded.require_hu()
    mask = decode_mask(roi_mask)
    if mask.shape != hu.shape:
        raise ValueError("roi_mask and CT slice must have the same dimensions")
    if not mask.any():
        raise ValueError("roi_mask must contain at least one foreground pixel")
    if pixel_spacing_row_column_mm is not None:
        spacing = tuple(pixel_spacing_row_column_mm)
        spacing_source = "provided"
    elif decoded.pixel_spacing_row_column_mm is not None:
        spacing = decoded.pixel_spacing_row_column_mm
        spacing_source = "dicom"
    else:
        spacing = (1.0, 1.0)
        spacing_source = "assumed"

    first_order = numpyradiomics.firstorder(
        hu,
        mask,
        voxelVolume=spacing[0] * spacing[1],
        binWidth=bin_width,
        extend=False,
    )
    first_order.pop("StandardDeviation", None)  # deprecated in PyRadiomics
    gldm = numpyradiomics.gldm(hu, mask, binWidth=bin_width)
    gldm.pop("GrayLevelNonUniformityNormalized", None)  # deprecated in PyRadiomics

    extracted: dict[str, float] = {}
    extracted.update(_prefixed("firstorder", first_order))
    extracted.update(
        _prefixed("glcm", numpyradiomics.glcm(hu, mask, binWidth=bin_width))
    )
    extracted.update(_prefixed("gldm", gldm))
    extracted.update(_prefixed("glrlm", _glrlm_2d(hu, mask, bin_width), _GLRLM_RENAMES))
    extracted.update(
        _prefixed(
            "glszm", numpyradiomics.glszm(hu, mask, binWidth=bin_width), _GLSZM_RENAMES
        )
    )
    extracted.update(
        _prefixed("ngtdm", numpyradiomics.ngtdm(hu, mask, binWidth=bin_width))
    )

    features = {
        name: value for name, value in extracted.items() if math.isfinite(value)
    }
    return {
        "intensity_units": "hu",
        "pixel_spacing_row_column_mm": list(spacing),
        "pixel_spacing_source": spacing_source,
        "bin_width": float(bin_width),
        "features": features,
    }
