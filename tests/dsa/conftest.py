"""Real coronary angiography example for the DSA tools."""

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes

# ARCADE sample (CC0) shipped with the Stenoz release: a 512-square X-ray
# coronary angiography frame with a cardiologist-annotated stenosis.
CORONARY_ANGIOGRAM_URL = (
    "https://raw.githubusercontent.com/uzbtrust/stenoz/main/samples/126.png"
)


@pytest.fixture(scope="session")
def coronary_angiogram() -> bytes:
    return _read_bytes(CORONARY_ANGIOGRAM_URL)
