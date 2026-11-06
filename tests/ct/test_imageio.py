"""Check CT decoding through public tools using real acquisitions."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.ct.classical import measure_ct_hu_regions, window_ct_slice

pytestmark = pytest.mark.slow


@pytest.fixture(params=["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"])
def ct_dicom(request: pytest.FixtureRequest) -> bytes:
    """Use both real lung acquisitions."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("center,width", [(40.0, 400.0), (0.0, 1.0)])
def test_dicom_window(ct_dicom: bytes, center: float, width: float) -> None:
    """Windowing uses calibrated HU, including the binary window case."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    hu = apply_modality_lut(dataset.pixel_array, dataset)
    output = window_ct_slice(ct_dicom, window_center=center, window_width=width)
    if width == 1:
        expected = np.where(hu <= center - 0.5, 0, 255)
    else:
        lower = center - 0.5 - (width - 1) / 2
        upper = center - 0.5 + (width - 1) / 2
        expected = np.interp(hu, [lower, upper], [0, 255]).round()
    with Image.open(BytesIO(output)) as image:
        np.testing.assert_array_equal(np.asarray(image), expected)


def test_dicom_geometry(ct_dicom: bytes) -> None:
    """Physical area uses the acquisition's pixel spacing."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    hu = apply_modality_lut(dataset.pixel_array, dataset)
    result = measure_ct_hu_regions(ct_dicom, float(hu.min()), float(hu.max()))
    (region,) = result["regions"]
    assert region["mean_hu"] == pytest.approx(float(hu.mean()))
    assert region["area_mm2"] == pytest.approx(
        hu.size * float(dataset.PixelSpacing[0]) * float(dataset.PixelSpacing[1])
    )


def test_calibrated_raster(ct_heart_image: bytes) -> None:
    """Explicit raster calibration is applied before HU measurements."""
    with Image.open(BytesIO(ct_heart_image)) as image:
        pixels = np.asarray(image).astype(np.float32)
    slope, intercept = 2.0, -1024.0
    hu = pixels * slope + intercept
    result = measure_ct_hu_regions(
        ct_heart_image,
        float(hu.min()),
        float(hu.max()),
        raster_units="hu",
        rescale_slope=slope,
        rescale_intercept=intercept,
    )
    (region,) = result["regions"]
    assert region["mean_hu"] == pytest.approx(float(hu.mean()))


def test_display_raster_requires_hu_calibration(ct_heart_image: bytes) -> None:
    """Display exports cannot silently be interpreted as HU."""
    with pytest.raises(ValueError, match="requires HU values"):
        window_ct_slice(ct_heart_image)
