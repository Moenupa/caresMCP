import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes


@pytest.fixture(scope="module")
def mlo_scattered_density() -> bytes:
    """Right MLO view of a breast with scattered fibroglandular density (BI-RADS B)."""
    return _read_bytes(
        "https://prod-images-static.radiopaedia.org/images/74726064/31de152c54fc2470634e5f97d4c54a641b0edd4d9018cd95e89140715da9f236.png"
    )


@pytest.fixture(scope="module")
def a_sample() -> bytes:
    return _read_bytes(
        "https://huggingface.co/MONAI/breast_density_classification/resolve/0.1.8/sample_data/A/sample_A1.jpg"
    )


@pytest.fixture(scope="module")
def b_sample() -> bytes:
    return _read_bytes(
        "https://huggingface.co/MONAI/breast_density_classification/resolve/0.1.8/sample_data/B/sample_B1.jpg"
    )


@pytest.fixture(scope="module")
def c_sample() -> bytes:
    return _read_bytes(
        "https://huggingface.co/MONAI/breast_density_classification/resolve/0.1.8/sample_data/C/sample_C1.jpg"
    )


@pytest.fixture(scope="module")
def d_sample() -> bytes:
    return _read_bytes(
        "https://huggingface.co/MONAI/breast_density_classification/resolve/0.1.8/sample_data/D/sample_D1.jpg"
    )
