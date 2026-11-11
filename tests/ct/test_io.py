"""Check CT image I/O helpers against real acquisitions."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.ct.io import (
    _decode_ct_slice,
    _decode_mask,
    _hu_range_mask,
    _model_image,
    _window_values,
)

pytestmark = pytest.mark.slow


@pytest.fixture(params=["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"])
def ct_dicom(request: pytest.FixtureRequest) -> bytes:
    """Use both real lung acquisitions."""
    return request.getfixturevalue(request.param)


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


def test_decode_image(ct_heart_image: bytes) -> None:
    """Display decoding and model preparation preserve the source pixels."""
    decoded = _decode_ct_slice(ct_heart_image)
    with Image.open(BytesIO(ct_heart_image)) as image:
        np.testing.assert_array_equal(decoded.values, np.asarray(image))
        np.testing.assert_array_equal(
            np.asarray(_model_image(decoded, center=40, width=400)),
            np.asarray(image.convert("RGB")),
        )
    with pytest.raises(ValueError, match="requires HU values"):
        decoded.require_hu()


def test_calibrated_image(ct_heart_image: bytes) -> None:
    """Explicit raster calibration applies to every pixel."""
    slope, intercept = 2.0, -1024.0
    decoded = _decode_ct_slice(
        ct_heart_image,
        raster_units="hu",
        rescale_slope=slope,
        rescale_intercept=intercept,
    )
    with Image.open(BytesIO(ct_heart_image)) as image:
        expected = np.asarray(image).astype(np.float32) * slope + intercept
    np.testing.assert_array_equal(decoded.require_hu(), expected)
    with pytest.raises(ValueError, match="Raster rescaling requires"):
        _decode_ct_slice(ct_heart_image, rescale_intercept=intercept)


@pytest.mark.parametrize("center,width", [(40.0, 400.0), (0.0, 1.0)])
def test_window_values(ct_dicom: bytes, center: float, width: float) -> None:
    """Windowing matches linear interpolation or a binary threshold."""
    hu = _decode_ct_slice(ct_dicom).require_hu()
    if width == 1:
        expected = np.where(hu <= center - 0.5, 0, 255)
    else:
        lower = center - 0.5 - (width - 1) / 2
        upper = center - 0.5 + (width - 1) / 2
        expected = np.interp(hu, [lower, upper], [0, 255]).round()
    np.testing.assert_array_equal(_window_values(hu, center, width), expected)


def test_hu_range_mask(ct_dicom: bytes) -> None:
    """Thresholds include pixels at both interval boundaries."""
    hu = _decode_ct_slice(ct_dicom).require_hu()
    minimum, maximum = np.quantile(hu, [0.25, 0.75], method="nearest")
    expected = np.logical_not((hu < minimum) | (hu > maximum))
    np.testing.assert_array_equal(_hu_range_mask(hu, minimum, maximum), expected)


def test_decode_mask(ct_heart_image: bytes) -> None:
    """Nonzero grayscale image pixels decode as foreground."""
    with Image.open(BytesIO(ct_heart_image)) as image:
        expected = np.asarray(image) != 0
    np.testing.assert_array_equal(_decode_mask(ct_heart_image, "mask"), expected)
