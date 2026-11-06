import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes


@pytest.fixture(scope="module")
def ct_heart_image() -> bytes:
    return _read_bytes(
        "https://prod-images-static.radiopaedia.org/images/69161224/dr-original.jpg"
    )


@pytest.fixture(scope="module")
def ct_lung_cancer_dicom() -> bytes:
    pytest.importorskip("pydicom")
    return _read_bytes(
        "https://saga-it.com/dicom/samples/files/ct-lung-screening-nlst/instance.dcm"
    )


@pytest.fixture(scope="module")
def ct_lung_nodule_dicom() -> bytes:
    pytest.importorskip("pydicom")
    return _read_bytes(
        "https://saga-it.com/dicom/samples/files/ct-chest-lidc-idri/instance.dcm"
    )
