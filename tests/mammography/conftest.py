"""Real mammograms shared by the mammography tool tests."""

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes

# Pin the model-card release so sample URLs cannot drift behind a moving tag.
MODEL_CARD_REVISION = "0.1.8"
MODEL_CARD_CATEGORIES = ("A", "B", "C", "D")
MODEL_CARD_SAMPLES_PER_CATEGORY = 1  # set to 4 for completeness


def _model_card_image(category: str, filename: str) -> str:
    """Resolve one labeled sample released with the MONAI model card."""
    return (
        "https://huggingface.co/MONAI/breast_density_classification/resolve/"
        f"{MODEL_CARD_REVISION}/sample_data/{category}/{filename}"
    )


@pytest.fixture(scope="session")
def mlo_scattered_density() -> bytes:
    """Right MLO view of a breast with scattered fibroglandular density (BI-RADS B)."""
    return _read_bytes(
        "https://prod-images-static.radiopaedia.org/images/74726064/31de152c54fc2470634e5f97d4c54a641b0edd4d9018cd95e89140715da9f236.png"
    )


@pytest.fixture(scope="session")
def model_card_samples() -> dict[str, bytes]:
    """Map each enabled model-card sample ID to its image bytes."""
    return {
        f"{category}{index}": _read_bytes(
            _model_card_image(category, f"sample_{category}{index}.jpg")
        )
        for category in MODEL_CARD_CATEGORIES
        for index in range(1, MODEL_CARD_SAMPLES_PER_CATEGORY + 1)
    }
