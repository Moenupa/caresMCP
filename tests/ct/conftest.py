"""Real CT inputs released by the pydicom and model projects."""

import hashlib

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes

# pydicom's released CQ500 head CT slice: 512x512, 14-bit stored values,
# HU rescaling, and axial PL orientation. Pinned to a pydicom-data commit.
PYDICOM_CT_REVISION = "abc42b90985fb6cf385aa4af766d2c9c94a257a4"
PYDICOM_CT_URL = (
    "https://cdn.jsdelivr.net/gh/pydicom/pydicom-data@"
    f"{PYDICOM_CT_REVISION}/data_store/data/693_UNCR.dcm"
)
PYDICOM_CT_SHA256 = "cc4cdd599231922ecf63de2ddacf03d51c4588805c9154c2eef1ff49c23b32be"

# Sample JPEGs published with the oohtmeel CT classifier. The model card states
# that a filename containing X is a negative cancer image and one containing Y
# is positive. Pinned to the checkpoint revision.
OOHTMEEL_REVISION = "780850a80eebf16528e6ad4cecec2d088478570b"
OOHTMEEL_RELEASE_ROOT = (
    "https://huggingface.co/oohtmeel/"
    "swin-tiny-patch4-finetuned-lung-cancer-ct-scans/resolve/"
    f"{OOHTMEEL_REVISION}/"
)
OOHTMEEL_NEGATIVE_JPEG = "_X000a109d-56da-4c3f-8680-55afa04d6ae0.dcm.jpg.jpg"
OOHTMEEL_POSITIVE_JPEG = "_Yffb25b49-7c7d-4dda-8cbf-88487698848a.dcm.jpg.jpg"
OOHTMEEL_SHA256 = {
    OOHTMEEL_NEGATIVE_JPEG: (
        "9dce9f4436fef2a88ee6ed15d6814c206a49f8c28f2f025800ab56c0b10b2462"
    ),
    OOHTMEEL_POSITIVE_JPEG: (
        "70f3a28477a31571b602f79c667e9585d729c6ff00eee03f22d38ac9aff9cc9b"
    ),
}


@pytest.fixture(scope="session")
def ct_dicom() -> bytes:
    """Return the released pydicom CT slice with HU calibration."""
    data = _read_bytes(PYDICOM_CT_URL)
    digest = hashlib.sha256(data).hexdigest()
    assert digest == PYDICOM_CT_SHA256, "pydicom CT release bytes changed"
    return data


def _read_released_jpeg(filename: str) -> bytes:
    data = _read_bytes(OOHTMEEL_RELEASE_ROOT + filename)
    digest = hashlib.sha256(data).hexdigest()
    assert digest == OOHTMEEL_SHA256[filename], f"{filename}: released bytes changed"
    return data


@pytest.fixture(scope="session")
def ct_released_negative() -> bytes:
    """Return the released sample the checkpoint author labels negative."""
    return _read_released_jpeg(OOHTMEEL_NEGATIVE_JPEG)


@pytest.fixture(scope="session")
def ct_released_positive() -> bytes:
    """Return the released sample the checkpoint author labels positive."""
    return _read_released_jpeg(OOHTMEEL_POSITIVE_JPEG)


@pytest.fixture(scope="session")
def ct_raster(ct_released_negative: bytes) -> bytes:
    """Return a released CT raster for display-image and calibration tests."""
    return ct_released_negative
