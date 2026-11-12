import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes

MODEL_CARD_REVISION = "0.1.8"


def _model_card_image(category: str, filename: str) -> str:
    """Resolve one labeled sample released with the MONAI model card."""
    return (
        "https://huggingface.co/MONAI/breast_density_classification/resolve/"
        f"{MODEL_CARD_REVISION}/sample_data/{category}/{filename}"
    )


@pytest.fixture(scope="module")
def mlo_scattered_density() -> bytes:
    """Right MLO view of a breast with scattered fibroglandular density (BI-RADS B)."""
    return _read_bytes(
        "https://prod-images-static.radiopaedia.org/images/74726064/31de152c54fc2470634e5f97d4c54a641b0edd4d9018cd95e89140715da9f236.png"
    )

@pytest.fixture(scope="module")
def b_sample() -> bytes:
    return _read_bytes(_model_card_image("B", "sample_B1.jpg"))


@pytest.fixture(scope="module")
def c_sample() -> bytes:
    return _read_bytes(_model_card_image("C", "sample_C2.jpg"))
