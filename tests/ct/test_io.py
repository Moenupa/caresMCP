"""Check CT image I/O helpers against released inputs and pydicom decoding."""

from io import BytesIO

import numpy as np
import pytest

from caresmcp.api.io import decode_as, decode_mask
from caresmcp.ct.io import (
    _decode_ct_slice,
    _model_image,
)

pytestmark = pytest.mark.slow


def test_decode_dicom(ct_dicom: bytes) -> None:
    """DICOM decoding retains calibrated pixels and acquisition geometry."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    decoded = _decode_ct_slice(ct_dicom)
    np.testing.assert_array_equal(
        decoded.require_hu(), apply_modality_lut(dataset.pixel_array, dataset)
    )
    assert decoded.pixel_spacing_row_column_mm == pytest.approx(dataset.PixelSpacing)
    assert decoded.image_orientation_patient == pytest.approx(
        dataset.ImageOrientationPatient
    )
    assert decoded.image_position_patient == pytest.approx(dataset.ImagePositionPatient)
    assert decoded.slice_thickness_mm == pytest.approx(dataset.SliceThickness)


def test_decode_image(ct_raster: bytes) -> None:
    """Display decoding and model preparation preserve the released pixels."""
    decoded = _decode_ct_slice(ct_raster)
    image = decode_as(ct_raster)
    np.testing.assert_array_equal(decoded.values, np.asarray(image))
    np.testing.assert_array_equal(
        np.asarray(_model_image(decoded, center=40, width=400)),
        np.asarray(image.convert("RGB")),
    )
    with pytest.raises(ValueError, match="requires HU values"):
        decoded.require_hu()


def test_calibrated_image(ct_raster: bytes) -> None:
    """Explicit raster calibration applies to every pixel."""
    slope, intercept = 2.0, -1024.0
    decoded = _decode_ct_slice(
        ct_raster,
        raster_units="hu",
        rescale_slope=slope,
        rescale_intercept=intercept,
    )
    image = decode_as(ct_raster)
    expected = np.asarray(image).astype(np.float32) * slope + intercept
    np.testing.assert_array_equal(decoded.require_hu(), expected)
    with pytest.raises(ValueError, match="Raster rescaling requires"):
        _decode_ct_slice(ct_raster, rescale_intercept=intercept)


def test_decode_mask(ct_raster: bytes) -> None:
    """Nonzero grayscale image pixels decode as foreground."""
    image = decode_as(ct_raster)
    expected = np.asarray(image) != 0
    np.testing.assert_array_equal(decode_mask(ct_raster), expected)
