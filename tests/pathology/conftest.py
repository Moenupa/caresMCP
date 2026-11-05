import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes

# pool to select from, we always use first one as test case
# samples are at 2x, 10x, 40x
SAMPLES = """
https://prod-images-static.radiopaedia.org/images/74884531/6f22f016116baa92e86b44a6bec617b490d615e54ebdbdc31078981193d6a675.png
https://prod-images-static.radiopaedia.org/images/74884532/8ed482a35146d868c83f7dd21128840039a515bbfa917c6831fdda93fbc3343f.png
https://prod-images-static.radiopaedia.org/images/74884533/4870726dffb5b8d84541c6e78c459f1051e33b30e4f689dfd33e6b2927621009.png
""".strip().splitlines()


@pytest.fixture(scope="session")
def pathology_image() -> bytes:
    """Return the first real pathology sample as image bytes."""
    return _read_bytes(SAMPLES[1])


@pytest.fixture(scope="session")
def pathology_images() -> list[bytes]:
    """Return all real pathology samples as image bytes."""
    return [_read_bytes(url) for url in SAMPLES]
