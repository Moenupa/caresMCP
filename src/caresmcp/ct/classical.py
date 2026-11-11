"""Classical CT preprocessing, segmentation, measurement, and registration tools."""

from io import BytesIO
from typing import Annotated, Any, Literal, Self

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from caresmcp.api.io import encode_as
from caresmcp.registry import CT, TASK_DEN, TASK_SEG

from .io import (
    CTInputFormat,
    CTRasterUnits,
    _decode_ct_slice,
    _decode_mask,
    _hu_range_mask,
    _window_values,
)

CTMotionModel = Literal["translation", "rigid", "affine"]
CTDenoiseMethod = Literal["bilateral", "gaussian", "median"]
CTDenoiseKernelSize = Literal[3, 5, 7, 9, 11, 13, 15]
CTImage = Annotated[
    bytes,
    Field(min_length=1, description="CT DICOM slice or grayscale raster."),
]
CTInputMode = Annotated[
    CTInputFormat,
    Field(description="CT input format."),
]
RasterRescaleSlope = Annotated[
    float,
    Field(allow_inf_nan=False, description="Raster rescale multiplier."),
]
RasterRescaleIntercept = Annotated[
    float,
    Field(allow_inf_nan=False, description="HU raster rescale offset."),
]
RasterUnits = Annotated[
    CTRasterUnits,
    Field(description="Use 'hu' only with known raster HU calibration."),
]
MinimumHU = Annotated[
    float,
    Field(allow_inf_nan=False, description="Inclusive minimum HU."),
]
MaximumHU = Annotated[
    float,
    Field(allow_inf_nan=False, description="Inclusive maximum HU."),
]
Mask = Annotated[bytes, Field(min_length=1, description="Binary mask image.")]
WindowCenter = Annotated[
    float, Field(allow_inf_nan=False, description="Window center in HU.")
]
WindowWidth = Annotated[
    float, Field(ge=1, allow_inf_nan=False, description="Window width in HU.")
]


class _HUInterval(BaseModel):
    minimum_hu: float
    maximum_hu: float

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.minimum_hu > self.maximum_hu:
            raise ValueError("minimum_hu must not exceed maximum_hu")
        return self


class _DenoiseOptions(BaseModel):
    method: CTDenoiseMethod
    kernel_size: CTDenoiseKernelSize

    @model_validator(mode="after")
    def validate_median_kernel(self) -> Self:
        if self.method == "median" and self.kernel_size > 5:
            raise ValueError(
                "median filtering supports kernel_size 3 or 5 for float HU data"
            )
        return self


@CT.register("window_ct_slice")
def window_ct_slice(
    image: CTImage,
    window_center: WindowCenter = 40.0,
    window_width: WindowWidth = 400.0,
    input_format: CTInputMode = "auto",
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    raster_units: RasterUnits = "display",
) -> bytes:
    """Render a single CT slice (in 8-bit PNG) with DICOM LINEAR HU window convention.

    Requires CT DICOM or a raster with known HU calibration. Display exports
    cannot recover the original HU values.
    """

    decoded = _decode_ct_slice(
        image, input_format, rescale_slope, rescale_intercept, raster_units
    )
    result = Image.fromarray(
        _window_values(decoded.require_hu(), window_center, window_width)
    )
    return encode_as(result)


@CT.register("segment_ct_hu_range")
@TASK_SEG.register("segment_ct_hu_range")
def segment_ct_hu_range(
    image: CTImage,
    minimum_hu: MinimumHU,
    maximum_hu: MaximumHU,
    input_format: CTInputMode = "auto",
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    closing_kernel_size: Annotated[
        int,
        Field(
            ge=0, description="Square closing width in pixels; zero disables closing."
        ),
    ] = 0,
    keep_largest_component: Annotated[
        bool,
        Field(description="Keep only the largest 8-connected foreground component."),
    ] = False,
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
        image, input_format, rescale_slope, rescale_intercept, raster_units
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
    image: CTImage,
    minimum_hu: MinimumHU,
    maximum_hu: MaximumHU,
    input_format: CTInputMode = "auto",
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    minimum_area_pixels: Annotated[
        int, Field(ge=1, description="Minimum connected component area in pixels.")
    ] = 1,
    maximum_regions: Annotated[
        int,
        Field(
            ge=1, description="Maximum regions returned, ordered by decreasing area."
        ),
    ] = 100,
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
        image, input_format, rescale_slope, rescale_intercept, raster_units
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


@CT.register("denoise_ct_slice")
@TASK_DEN.register("denoise_ct_slice")
def denoise_ct_slice(
    image: CTImage,
    method: Annotated[
        CTDenoiseMethod,
        Field(description="Classical denoising filter."),
    ] = "bilateral",
    input_format: CTInputMode = "auto",
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    kernel_size: Annotated[
        CTDenoiseKernelSize,
        Field(description="Odd filter width in pixels; median supports only 3 or 5."),
    ] = 5,
    sigma_hu: Annotated[
        float,
        Field(
            gt=0,
            allow_inf_nan=False,
            description="Bilateral intensity scale in HU or raster value units.",
        ),
    ] = 50.0,
    sigma_pixels: Annotated[
        float,
        Field(
            gt=0,
            allow_inf_nan=False,
            description="Bilateral spatial scale or Gaussian standard deviation.",
        ),
    ] = 1.5,
    raster_units: RasterUnits = "display",
) -> bytes:
    """Denoise one CT slice with a classical edge-aware or local filter.

    The output is a 32-bit floating-point grayscale TIFF so pixel values remain
    in HU (for DICOM input) or in the raster values after the supplied rescale.
    Geometry and intensity-unit metadata are not retained. To reuse an HU
    output, set raster_units='hu' with slope 1 and intercept 0.
    These filters are research baselines and can alter small structures.

    Returns:
        A 32-bit floating-point TIFF containing filtered values in the input
        intensity units.
    """

    _DenoiseOptions(method=method, kernel_size=kernel_size)
    decoded = _decode_ct_slice(
        image, input_format, rescale_slope, rescale_intercept, raster_units
    )
    if method == "bilateral":
        denoised = cv2.bilateralFilter(
            decoded.values, kernel_size, sigmaColor=sigma_hu, sigmaSpace=sigma_pixels
        )
    elif method == "gaussian":
        denoised = cv2.GaussianBlur(
            decoded.values, (kernel_size, kernel_size), sigmaX=sigma_pixels
        )
    else:
        denoised = cv2.medianBlur(decoded.values, kernel_size)
    output = BytesIO()
    Image.fromarray(np.asarray(denoised, dtype=np.float32)).save(output, format="TIFF")
    return output.getvalue()


@CT.register("register_ct_slices")
def register_ct_slices(
    fixed_image: Annotated[
        bytes, Field(min_length=1, description="Reference CT slice bytes.")
    ],
    moving_image: Annotated[
        bytes,
        Field(min_length=1, description="CT slice bytes to align to the reference."),
    ],
    motion_model: Annotated[
        CTMotionModel,
        Field(
            description="Translation, rigid rotation plus translation, or affine transform."
        ),
    ] = "rigid",
    input_format: CTInputMode = "auto",
    fixed_rescale_slope: Annotated[
        float,
        Field(
            allow_inf_nan=False,
            description="Stored-value multiplier for the fixed raster.",
        ),
    ] = 1.0,
    fixed_rescale_intercept: Annotated[
        float,
        Field(
            allow_inf_nan=False, description="Stored-value offset for the fixed raster."
        ),
    ] = 0.0,
    moving_rescale_slope: Annotated[
        float,
        Field(
            allow_inf_nan=False,
            description="Stored-value multiplier for the moving raster.",
        ),
    ] = 1.0,
    moving_rescale_intercept: Annotated[
        float,
        Field(
            allow_inf_nan=False,
            description="Stored-value offset for the moving raster.",
        ),
    ] = 0.0,
    window_center: Annotated[
        float,
        Field(
            allow_inf_nan=False, description="HU center used to normalize both images."
        ),
    ] = 40.0,
    window_width: Annotated[
        float,
        Field(
            ge=1,
            allow_inf_nan=False,
            description="HU width used to normalize both images.",
        ),
    ] = 400.0,
    maximum_iterations: Annotated[
        int, Field(ge=1, description="Maximum ECC optimization iterations.")
    ] = 500,
    convergence_epsilon: Annotated[
        float,
        Field(
            gt=0,
            allow_inf_nan=False,
            description="Stop when correlation changes by less than this value.",
        ),
    ] = 1e-6,
    raster_units: RasterUnits = "display",
) -> dict[str, Any]:
    """Estimate a two-dimensional transform aligning one CT slice to another.

    The returned matrix maps moving-image coordinates to fixed-image
    coordinates and can be used with OpenCV warpAffine without inverse
    warping. HU inputs are windowed; display inputs retain their intensities.
    Inputs must have matching dimensions and similar anatomy with a small
    initial displacement. Spacing and orientation metadata are not used to
    resample the images. A high correlation alone does not establish anatomical
    alignment; inspect the transformed image before using the result.

    Returns:
        Enhanced correlation coefficient and a 2-by-3 moving-to-fixed matrix.
    """

    fixed_slice = _decode_ct_slice(
        fixed_image,
        input_format,
        fixed_rescale_slope,
        fixed_rescale_intercept,
        raster_units,
    )
    moving_slice = _decode_ct_slice(
        moving_image,
        input_format,
        moving_rescale_slope,
        moving_rescale_intercept,
        raster_units,
    )
    if fixed_slice.values.shape != moving_slice.values.shape:
        raise ValueError("fixed_image and moving_image must have the same dimensions")
    if fixed_slice.intensity_units != moving_slice.intensity_units:
        raise ValueError(
            "fixed_image and moving_image must use the same intensity units"
        )
    if fixed_slice.intensity_units == "hu":
        fixed = _window_values(fixed_slice.values, window_center, window_width)
        moving = _window_values(moving_slice.values, window_center, window_width)
    else:
        fixed = fixed_slice.values
        moving = moving_slice.values
    fixed = fixed.astype(np.float32) / 255
    moving = moving.astype(np.float32) / 255
    motion_types = {
        "translation": cv2.MOTION_TRANSLATION,
        "rigid": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        maximum_iterations,
        convergence_epsilon,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            fixed,
            moving,
            warp,
            motion_types[motion_model],
            criteria,
        )
    except cv2.error as exc:
        raise ValueError("CT slice registration did not converge") from exc
    if not np.isfinite(correlation) or not np.isfinite(warp).all():
        raise ValueError("CT slice registration returned non-finite results")
    if abs(float(np.linalg.det(warp[:, :2]))) < 1e-8:
        raise ValueError("CT slice registration returned a singular transform")
    warp = cv2.invertAffineTransform(warp)
    return {
        "motion_model": motion_model,
        "correlation": float(correlation),
        "moving_to_fixed_matrix": [
            [float(value) for value in row] for row in warp.tolist()
        ],
    }


@CT.register("evaluate_ct_segmentation")
def evaluate_ct_segmentation(
    predicted_mask: Mask,
    ground_truth_mask: Mask,
) -> dict[str, float]:
    """Compare a CT segmentation mask with a reference binary mask.

    Nonzero pixels are foreground; fully transparent pixels are background.
    Metrics range from zero to one, with one indicating perfect agreement.
    Precision measures the fraction of predicted pixels matching the reference;
    recall measures the fraction of reference pixels recovered.
    Masks must have matching dimensions and a nonempty reference foreground.
    An empty prediction returns zero for every metric.
    """

    predicted = _decode_mask(predicted_mask, "predicted_mask")
    reference = _decode_mask(ground_truth_mask, "ground_truth_mask")
    if predicted.shape != reference.shape:
        raise ValueError("predicted_mask and ground_truth_mask must have the same size")
    reference_count = int(reference.sum())
    if reference_count == 0:
        raise ValueError("ground_truth_mask must contain a foreground pixel")
    predicted_count = int(predicted.sum())
    intersection = int(np.logical_and(predicted, reference).sum())
    union = predicted_count + reference_count - intersection
    return {
        "dice": float(2 * intersection / (predicted_count + reference_count)),
        "intersection_over_union": float(intersection / union),
        "precision": float(intersection / predicted_count) if predicted_count else 0.0,
        "recall": float(intersection / reference_count),
    }
