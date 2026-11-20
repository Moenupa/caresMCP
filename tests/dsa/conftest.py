"""Official Stenoz coronary angiography samples for the DSA tools."""

import json

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes

# The Stenoz release (MIT) ships ten ARCADE frames with cardiologist stenosis
# boxes. Pin the repository revision so a moved branch cannot change the inputs.
STENOZ_REVISION = "2c3df091c8ff7b035ec925fcd496c6402e6de8a3"
STENOZ_SAMPLES_ROOT = (
    f"https://cdn.jsdelivr.net/gh/uzbtrust/stenoz@{STENOZ_REVISION[:9]}/samples/"
)
CORONARY_SAMPLE_NAMES = (
    "104",
    "110",
    "126",
    "131",
    "141",
    "143",
    "148",
    "159",
    "167",
    "21",
)


@pytest.fixture(scope="session")
def coronary_samples() -> dict[str, bytes]:
    """Read every released sample frame from the revision-pinned snapshot."""
    return {
        name: _read_bytes(f"{STENOZ_SAMPLES_ROOT}{name}.png")
        for name in CORONARY_SAMPLE_NAMES
    }


@pytest.fixture(scope="session")
def coronary_annotations() -> dict[str, dict[str, list[list[float]]]]:
    """Read the released cardiologist boxes keyed by sample file name."""
    return json.loads(_read_bytes(f"{STENOZ_SAMPLES_ROOT}annotations.json"))
