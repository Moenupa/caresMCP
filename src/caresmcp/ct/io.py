"""Package-internal CT decoding and rendering shared by public tool interfaces."""

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from caresmcp.api.io import (
    DicomImage,
    decode_as,
    decode_dicom,
    is_dicom,
    is_grayscale,
)

CTRasterUnits = Literal["display", "hu"]


@dataclass(frozen=True)
class DecodedCTSlice:
    """A finite, nonempty 2D float32 slice with optional DICOM geometry.

    Fields are frozen; the underlying NumPy array remains mutable.
    """

    values: NDArray[np.float32]
    intensity_units: CTRasterUnits
    pixel_spacing_row_column_mm: tuple[float, float] | None = None
    image_orientation_patient: tuple[float, ...] | None = None
    image_position_patient: tuple[float, ...] | None = None
    slice_thickness_mm: float | None = None

    def require_hu(self) -> NDArray[np.float32]:
        if self.intensity_units != "hu":
            raise ValueError(
                "This tool requires HU values: use CT DICOM or set raster_units='hu' "
                "only for a raster with known HU calibration; display JPEG values "
                "cannot recover the original HU"
            )
        return self.values


def _parse_rescale_parameters(slope: float, intercept: float) -> tuple[float, float]:
    try:
        slope = float(slope)
        intercept = float(intercept)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rescale_slope and rescale_intercept must be finite") from exc
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("rescale_slope and rescale_intercept must be finite")

    return slope, intercept


def _validate_slice_pixels(pixels: NDArray[Any]) -> NDArray[np.float32]:
    if pixels.ndim != 2 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
        raise ValueError("CT input must contain one non-empty two-dimensional slice")
    pixels = pixels.astype(np.float32, copy=False)
    if not np.isfinite(pixels).all():
        raise ValueError("CT slice must contain only finite pixel values")
    return pixels


def _validate_dicom_calibration(dicom: DicomImage) -> None:
    if dicom.modality != "CT" or str(dicom.rescale_type or "HU").upper() != "HU":
        raise ValueError("DICOM input must contain CT values in HU")
    if dicom.has_modality_lut:
        if str(dicom.modality_lut_type or "").upper() != "HU":
            raise ValueError("DICOM modality LUT must declare HU output")
    elif dicom.rescale_slope is None or dicom.rescale_intercept is None:
        raise ValueError("DICOM CT input requires an HU calibration")
    else:
        slope = dicom.rescale_slope
        intercept = dicom.rescale_intercept
        if not math.isfinite(slope) or slope == 0 or not math.isfinite(intercept):
            raise ValueError("DICOM rescale calibration must be finite and nonzero")


def _decode_dicom(image: bytes) -> DecodedCTSlice:
    dicom = decode_dicom(image, modality_lut=True)
    _validate_dicom_calibration(dicom)
    if not dicom.is_monochrome or dicom.number_of_frames != 1:
        raise ValueError("CT input must contain one monochrome DICOM frame")
    if dicom.modality_values is None:
        raise RuntimeError("DICOM modality values were not decoded")
    return DecodedCTSlice(
        values=_validate_slice_pixels(dicom.modality_values[0]),
        intensity_units="hu",
        pixel_spacing_row_column_mm=dicom.pixel_spacing_row_column_mm,
        image_orientation_patient=dicom.image_orientation_patient,
        image_position_patient=dicom.image_position_patient,
        slice_thickness_mm=dicom.slice_thickness_mm,
    )


def _read_grayscale_raster(image: bytes) -> NDArray[Any]:
    try:
        source = decode_as(image)
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("CT input must contain exactly one raster frame")
        if not is_grayscale(source.mode):
            raise ValueError("CT slice image must be single-channel grayscale")
        pixels = np.asarray(source)
    except OSError as exc:
        raise ValueError("image must contain a readable CT slice image") from exc
    return pixels


def _decode_raster(
    image: bytes, slope: float, intercept: float, units: CTRasterUnits
) -> DecodedCTSlice:
    if units == "display" and (slope != 1.0 or intercept != 0.0):
        raise ValueError("Raster rescaling requires raster_units='hu'")
    pixels = _read_grayscale_raster(image).astype(np.float32) * slope + intercept
    return DecodedCTSlice(_validate_slice_pixels(pixels), units)


def _decode_ct_slice(
    image: bytes,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    raster_units: CTRasterUnits = "display",
) -> DecodedCTSlice:
    if is_dicom(image):
        return _decode_dicom(image)
    slope, intercept = _parse_rescale_parameters(rescale_slope, rescale_intercept)
    return _decode_raster(image, slope, intercept, raster_units)


def _mask_foreground(source: Image.Image, name: str) -> NDArray[np.bool_]:
    if is_grayscale(source.mode):
        values = np.asarray(source)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite mask values")
        return values != 0
    rgba = np.asarray(source.convert("RGBA"))
    return np.any(rgba[..., :3] != 0, axis=-1) & (rgba[..., 3] != 0)


def _decode_mask(mask: bytes, name: str) -> NDArray[np.bool_]:
    return _mask_foreground(decode_as(mask), name)


def _model_image(slice: DecodedCTSlice, center: float, width: float) -> Image.Image:
    """Prepare RGB model input, preserving already rendered display values."""
    if slice.intensity_units == "hu":
        pixels = _window_values(slice.values, center, width)
    else:
        if slice.values.min() < 0 or slice.values.max() > 255:
            raise ValueError("Display model inputs must have intensities from 0 to 255")
        pixels = np.rint(slice.values).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def _window_values(
    hu: NDArray[np.float32], center: float, width: float
) -> NDArray[np.uint8]:
    """Render HU values to 8-bit grayscale using a linear CT window."""

    if not math.isfinite(center):
        raise ValueError("window_center must be finite")
    if not math.isfinite(width) or width < 1:
        raise ValueError("window_width must be at least 1")
    if width == 1:
        return (hu > center - 0.5).astype(np.uint8) * 255
    scaled = (
        (np.asarray(hu, dtype=np.float64) - (center - 0.5)) / (width - 1) + 0.5
    ) * 255
    return np.rint(np.clip(scaled, 0, 255)).astype(np.uint8)


def _hu_range_mask(
    hu: NDArray[np.float32], minimum_hu: float, maximum_hu: float
) -> NDArray[np.bool_]:
    return (hu >= minimum_hu) & (hu <= maximum_hu)
