"""Test DICOM decoding against real clinical and sample images."""

from io import BytesIO

import numpy as np
import pydicom
import pytest
from pydicom.pixels import apply_modality_lut

from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import decode_dicom, is_dicom

pytestmark = pytest.mark.slow

CT_DICOM_URL = (
    "https://saga-it.com/dicom/samples/files/ct-lung-screening-nlst/instance.dcm"
)
CHEST_DICOM_URL = (
    "https://raw.githubusercontent.com/mlmed/torchxrayvision/1.5.7/tests/"
    "1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm"
)
MULTIFRAME_DICOM_URL = (
    "https://raw.githubusercontent.com/pydicom/pydicom/v3.0.1/"
    "src/pydicom/data/test_files/SC_rgb_rle_2frame.dcm"
)


@pytest.fixture(scope="module")
def ct_dicom() -> bytes:
    """A real CT slice with HU rescaling and acquisition geometry."""
    return _read_bytes(CT_DICOM_URL)


@pytest.fixture(scope="module")
def chest_dicom() -> bytes:
    """A real 1024x1024 PA chest radiograph DICOM."""
    return _read_bytes(CHEST_DICOM_URL)


@pytest.fixture(scope="module")
def multiframe_dicom() -> bytes:
    """A real two-frame RGB secondary-capture DICOM with RLE pixel data."""
    return _read_bytes(MULTIFRAME_DICOM_URL)


def test_is_dicom(ct_dicom: bytes, chest_dicom: bytes, dummy_jpeg: bytes) -> None:
    """Recognize Part 10 files and reject raster images."""
    assert is_dicom(ct_dicom)
    assert is_dicom(chest_dicom)
    assert not is_dicom(dummy_jpeg)


def test_decode_ct_slice(ct_dicom: bytes) -> None:
    """Return stored pixels and the acquisition metadata CT tools depend on."""
    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    decoded = decode_dicom(ct_dicom)
    assert decoded.pixels.shape == (1, *dataset.pixel_array.shape)
    np.testing.assert_array_equal(decoded.pixels[0], dataset.pixel_array)
    assert decoded.modality_values is None
    assert decoded.modality == "CT"
    assert decoded.number_of_frames == 1
    assert decoded.is_monochrome
    assert decoded.bits_stored == dataset.BitsStored
    assert decoded.rescale_slope == float(dataset.RescaleSlope)
    assert decoded.rescale_intercept == float(dataset.RescaleIntercept)
    assert not decoded.has_modality_lut
    assert decoded.pixel_spacing_row_column_mm == pytest.approx(dataset.PixelSpacing)
    assert decoded.image_orientation_patient == pytest.approx(
        dataset.ImageOrientationPatient
    )
    assert decoded.image_position_patient == pytest.approx(dataset.ImagePositionPatient)
    assert decoded.slice_thickness_mm == pytest.approx(dataset.SliceThickness)


def test_decode_ct_modality_lut(ct_dicom: bytes) -> None:
    """Apply the modality LUT exactly as pydicom does when requested."""
    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    decoded = decode_dicom(ct_dicom, modality_lut=True)
    assert decoded.modality_values is not None
    np.testing.assert_array_equal(
        decoded.modality_values[0], apply_modality_lut(dataset.pixel_array, dataset)
    )


def test_decode_radiograph(chest_dicom: bytes) -> None:
    """Return one MONOCHROME2 frame with its declared bit depth."""
    dataset = pydicom.dcmread(BytesIO(chest_dicom))
    decoded = decode_dicom(chest_dicom)
    assert decoded.pixels.shape == (1, 1024, 1024)
    np.testing.assert_array_equal(decoded.monochrome_frame(), dataset.pixel_array)
    assert decoded.modality == "CR"
    assert decoded.photometric_interpretation == "MONOCHROME2"
    assert decoded.bits_stored == dataset.BitsStored
    with pytest.raises(ValueError, match="index is out of range"):
        decoded.monochrome_frame(index=1)


def test_decode_multiframe_color(multiframe_dicom: bytes) -> None:
    """Keep every color frame and expose the normalized pixel layout."""
    decoded = decode_dicom(multiframe_dicom)
    assert decoded.pixels.shape == (2, 100, 100, 3)
    assert decoded.number_of_frames == 2
    assert not decoded.is_monochrome
    assert decoded.modality_values is None
    with pytest.raises(ValueError, match="must be monochrome"):
        decoded.monochrome_frame()


def test_rejects_non_dicom(dummy_jpeg: bytes) -> None:
    """Reject bytes without a DICOM Part 10 header."""
    with pytest.raises(ValueError, match="not a readable DICOM image"):
        decode_dicom(dummy_jpeg)
