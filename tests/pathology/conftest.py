import pytest

from caresmcp.api.decorators import download_bytes

# CONCH is the only pathology model in this package whose official repository
# ships an input tile. ``docs/roi1.jpg`` is the H&E image used by every CONCH
# notebook (basics usage, zero-shot starter, and retrieval) and is therefore
# the shared official pathology sample for this test suite.
CONCH_REPOSITORY = "Mahmoodlab/CONCH"
CONCH_REVISION = "141cc09c7d4ff33d8eda562bd75169b457f71a62"
CONCH_DEMO_URL = (
    f"https://cdn.jsdelivr.net/gh/{CONCH_REPOSITORY}@{CONCH_REVISION}/docs/roi1.jpg"
)
CONCH_DEMO_SHA256 = "bea865947951babe8418a53c932f8ec856fdd86794a59c8e8ea4fdfedaab46e5"


@pytest.fixture(scope="session")
def pathology_image() -> bytes:
    """Return CONCH's official demo H&E tile as image bytes."""
    return download_bytes(CONCH_DEMO_URL, checksum=CONCH_DEMO_SHA256)
