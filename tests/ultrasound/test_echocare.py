"""Integration coverage for EchoCare using real ultrasound images."""

import math
import os
from pathlib import Path

import pytest

from caresmcp.registry import TASK_EMBED, ULTRASOUND
from caresmcp.ultrasound.echocare import echocare_image_embed

SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data_subset10" / "ultrasound"
SAMPLE_IMAGE_PATHS = (
    tuple(
        sorted(
            path
            for path in SAMPLE_DIRECTORY.rglob("*")
            if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
    )
    if SAMPLE_DIRECTORY.exists()
    else ()
)


def test_echocare_tool_is_registered() -> None:
    """Expose EchoCare image embeddings through both catalogues."""
    assert "echocare_image_embed" in ULTRASOUND.get_all()
    assert "echocare_image_embed" in TASK_EMBED.get_all()


@pytest.mark.slow
def test_echocare_with_real_ultrasound_image() -> None:
    """Extract an embedding from a real example when weights are available."""
    checkpoint_path = os.environ.get("ECHOCARE_CHECKPOINT", "")
    if not SAMPLE_IMAGE_PATHS:
        pytest.skip("data_subset10/ultrasound samples are not available")
    if not checkpoint_path:
        pytest.skip("set ECHOCARE_CHECKPOINT to the downloaded encoder weights")

    try:
        result = echocare_image_embed(
            SAMPLE_IMAGE_PATHS[0].read_bytes(), checkpoint_path=checkpoint_path
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    assert result["model"] == "CAIR-HKISI/EchoCare"
    assert len(result["embedding"]) == 2048
    assert all(math.isfinite(value) for value in result["embedding"])
