"""Classical CT preprocessing, segmentation, and measurement tools."""

from typing import Any, Self

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, model_validator

from caresmcp.api.io import encode_as
from caresmcp.registry import CT, TASK_SEG

from ._arg import (
    CTImage,
    MaxHU,
    MeasureCtHuRegionsMaximumRegionsArg,
    MeasureCtHuRegionsMinimumAreaPixelsArg,
    MinHU,
    RasterRescaleIntercept,
    RasterRescaleSlope,
    RasterUnits,
    SegmentCtHuRangeClosingKernelSizeArg,
    SegmentCtHuRangeKeepLargestComponentArg,
    WindowCenter,
    WindowWidth,
)
from .io import (
    _decode_ct_slice,
    _hu_range_mask,
    _window_values,
)


class _HUInterval(BaseModel):
    minimum_hu: float
    maximum_hu: float

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.minimum_hu > self.maximum_hu:
            raise ValueError("minimum_hu must not exceed maximum_hu")
        return self


@CT.register("window_ct_slice")
def window_ct_slice(
    image_or_dicom: CTImage,
    window_center: WindowCenter = 40.0,
    window_width: WindowWidth = 400.0,
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    raster_units: RasterUnits = "display",
) -> bytes:
    """Render a single CT slice (in 8-bit PNG) with DICOM LINEAR HU window convention.

    Requires CT DICOM or a raster with known HU calibration. Display exports
    cannot recover the original HU values.
    """

    decoded = _decode_ct_slice(
        image_or_dicom, rescale_slope, rescale_intercept, raster_units
    )
    result = Image.fromarray(
        _window_values(decoded.require_hu(), window_center, window_width)
    )
    return encode_as(result)


@CT.register("segment_ct_hu_range")
@TASK_SEG.register("segment_ct_hu_range")
def segment_ct_hu_range(
    image_or_dicom: CTImage,
    minimum_hu: MinHU,
    maximum_hu: MaxHU,
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    closing_kernel_size: SegmentCtHuRangeClosingKernelSizeArg = 0,
    keep_largest_component: SegmentCtHuRangeKeepLargestComponentArg = False,
    raster_units: RasterUnits = "display",
) -> bytes:
    """Create a binary mask (in 8-bit PNG) for pixels within a supplied HU interval.

    White (255) is selected foreground; black (0) is background. Closing fills
    small gaps before optional largest-component selection.
    The threshold is a reproducible intensity baseline. It does not identify
    an organ or lesion without target-specific threshold selection.
    """

    _HUInterval(minimum_hu=minimum_hu, maximum_hu=maximum_hu)
    decoded = _decode_ct_slice(
        image_or_dicom, rescale_slope, rescale_intercept, raster_units
    )
    mask = (
        np.asarray(
            _hu_range_mask(decoded.require_hu(), minimum_hu, maximum_hu), dtype=np.uint8
        )
        * 255
    )
    if closing_kernel_size > 1:
        kernel = np.ones((closing_kernel_size, closing_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if keep_largest_component:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if component_count > 1:
            largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            mask = np.asarray(labels == largest, dtype=np.uint8) * 255
    return encode_as(Image.fromarray(mask))


@CT.register("measure_ct_hu_regions")
def measure_ct_hu_regions(
    image_or_dicom: CTImage,
    minimum_hu: MinHU,
    maximum_hu: MaxHU,
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    minimum_area_pixels: MeasureCtHuRegionsMinimumAreaPixelsArg = 1,
    maximum_regions: MeasureCtHuRegionsMaximumRegionsArg = 100,
    raster_units: RasterUnits = "display",
) -> dict[str, Any]:
    """Measure connected components selected by a CT HU interval.

    Measurements use 8-connected two-dimensional regions. XY coordinates
    are in pixels, with the origin at the top-left. Physical area is included
    only when DICOM pixel spacing is present. The region count is measured
    after the minimum-area filter and before the maximum-regions limit;
    regions_truncated indicates that additional qualifying regions were omitted.

    Returns:
        HU thresholds, available DICOM geometry, and region bounding boxes,
        centroids, areas, and mean HU values.
    """

    _HUInterval(minimum_hu=minimum_hu, maximum_hu=maximum_hu)
    decoded = _decode_ct_slice(
        image_or_dicom, rescale_slope, rescale_intercept, raster_units
    )
    hu = decoded.require_hu()
    binary = np.asarray(_hu_range_mask(hu, minimum_hu, maximum_hu), dtype=np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    candidates = [
        label
        for label in range(1, component_count)
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area_pixels
    ]
    candidates.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    selected = candidates[:maximum_regions]
    sums = np.bincount(
        np.asarray(labels, dtype=np.intp).ravel(),
        weights=hu.ravel(),
        minlength=component_count,
    )
    spacing = decoded.pixel_spacing_row_column_mm
    regions: list[dict[str, Any]] = []
    for label in selected:
        x, y, width, height, area = (int(value) for value in stats[label])
        region: dict[str, Any] = {
            "label": label,
            "area_pixels": area,
            "bounding_box_xywh": [x, y, width, height],
            "centroid_xy": [float(value) for value in centroids[label]],
            "mean_hu": float(sums[label] / area),
        }
        if spacing is not None:
            region["area_mm2"] = float(area * spacing[0] * spacing[1])
        regions.append(region)
    return {
        "intensity_units": "hu",
        "minimum_hu": float(minimum_hu),
        "maximum_hu": float(maximum_hu),
        "pixel_spacing_row_column_mm": list(spacing) if spacing is not None else None,
        "image_orientation_patient": (
            list(decoded.image_orientation_patient)
            if decoded.image_orientation_patient is not None
            else None
        ),
        "image_position_patient": (
            list(decoded.image_position_patient)
            if decoded.image_position_patient is not None
            else None
        ),
        "slice_thickness_mm": decoded.slice_thickness_mm,
        "image_size_xy": [int(hu.shape[1]), int(hu.shape[0])],
        "qualifying_region_count": len(candidates),
        "regions_truncated": len(candidates) > len(selected),
        "regions": regions,
    }
