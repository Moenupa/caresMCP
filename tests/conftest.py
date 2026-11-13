import os

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes


@pytest.fixture(scope="session")
def dummy_jpeg():
    return _read_bytes("https://placehold.co/128x128.jpg")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Modify test collection based on markers and environment."""
    _handle_slow_tests(items)
    _handle_cuda_tests(items)


def _handle_slow_tests(items: list[pytest.Item]):
    """Skip slow tests unless RUN_SLOW is enabled."""
    if os.environ.get("RUN_SLOW") == "1":
        return

    skip_slow = pytest.mark.skip(reason="slow test (set RUN_SLOW=1 to run)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _handle_cuda_tests(items: list[pytest.Item]):
    """Skip CUDA tests unless a CUDA device is available."""
    from caresmcp.constants import DEVICE

    if DEVICE == "cuda":
        return

    skip_cuda = pytest.mark.skip(reason="cuda test (require a cuda device to run)")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
