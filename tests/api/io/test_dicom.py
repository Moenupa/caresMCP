"""Test DICOM decoding against real clinical and sample images."""

from io import BytesIO

import numpy as np
import pydicom
import pytest
from pydicom.pixels import apply_modality_lut

from caresmcp.api.decorators import download_bytes
from caresmcp.api.io import decode_dicom, is_dicom

pytestmark = pytest.mark.slow

# pydicom's released CQ500 head CT slice: 512x512, HU rescaling, axial PL.
# Pinned to a pydicom-data commit, the same release the CT tool tests use.
CT_DICOM_URL = (
    "https://cdn.jsdelivr.net/gh/pydicom/pydicom-data@"
    "abc42b90985fb6cf385aa4af766d2c9c94a257a4/data_store/data/693_UNCR.dcm"
)
CT_DICOM_SHA256 = "cc4cdd599231922ecf63de2ddacf03d51c4588805c9154c2eef1ff49c23b32be"
# The 1024x1024 PA chest radiograph shipped in TorchXRayVision's tests at the
# 1.5.7 tag. jsDelivr mirrors the same Git blob more reliably than
# raw.githubusercontent.com.
CHEST_DICOM_URL = (
    "https://cdn.jsdelivr.net/gh/mlmed/torchxrayvision@1.5.7/tests/"
    "1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm"
)
CHEST_DICOM_SHA256 = "13d378bfe1af0724e8cbe51adde36882cfdc6a69361c8a48ac30f688c3af7613"
# pydicom's released two-frame RGB secondary capture with RLE pixel data at
# the v3.0.1 tag.
MULTIFRAME_DICOM_URL = (
    "https://cdn.jsdelivr.net/gh/pydicom/pydicom@v3.0.1/"
    "src/pydicom/data/test_files/SC_rgb_rle_2frame.dcm"
)
MULTIFRAME_DICOM_SHA256 = (
    "cc9cd098ab099b5f7a18c4599f2858d2f3f3471590ff8a14d4cf7c834692d9f0"
)


@pytest.fixture(scope="module")
def ct_dicom() -> bytes:
    """A real CT slice with HU rescaling and acquisition geometry."""
    return download_bytes(CT_DICOM_URL, checksum=CT_DICOM_SHA256)


@pytest.fixture(scope="module")
def chest_dicom() -> bytes:
    """A real 1024x1024 PA chest radiograph DICOM."""
    return download_bytes(CHEST_DICOM_URL, checksum=CHEST_DICOM_SHA256)


@pytest.fixture(scope="module")
def multiframe_dicom() -> bytes:
    """A real two-frame RGB secondary-capture DICOM with RLE pixel data."""
    return download_bytes(MULTIFRAME_DICOM_URL, checksum=MULTIFRAME_DICOM_SHA256)


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
