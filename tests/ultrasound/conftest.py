import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes


@pytest.fixture(scope="session")
def ultrasound_image() -> bytes:
    return _read_bytes(
        "https://prod-images-static.radiopaedia.org/images/75936103/66061f99ceed45ca01fd96c37182847f60d0f915e419b06ed5492510e4b6a0d8_big_gallery.jpeg"
    )


@pytest.fixture(scope="session")
def ultrasound_video() -> bytes:
    return _read_bytes(
        "https://prod-images-static.radiopaedia.org/images/75936110/94f539f7b8fef0a1695c4c25f03072f6768c8a580c03927f11121d127c7fe8a0.mp4"
    )
