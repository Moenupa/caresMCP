import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes


@pytest.fixture(scope="module", autouse=True)
def cxr_image() -> bytes:
    # a hernia case
    return _read_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_000.png"
    )
