"""EchoNet-Dynamic function coverage using the shared PanEcho samples."""

import math
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.registry import TASK_SEG, ULTRASOUND
from caresmcp.ultrasound.echonet_dynamic import (
    _validate_channel_statistics,
    echonet_dynamic_predict_ef,
    echonet_dynamic_segment_lv,
)


@pytest.fixture(scope="session")
def channel_statistics(
    ultrasound_image: bytes,
) -> tuple[list[float], list[float]]:
    """Compute sample RGB statistics for smoke tests, not accuracy evaluation."""
    with Image.open(BytesIO(ultrasound_image)) as source:
        pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    return pixels.mean(axis=(0, 1)).tolist(), pixels.std(axis=(0, 1)).tolist()


def test_echonet_dynamic_tools_are_registered() -> None:
    """Expose EF prediction and LV segmentation through their catalogues."""
    assert "echonet_dynamic_predict_ef" in ULTRASOUND.get_all()
    assert "echonet_dynamic_segment_lv" in ULTRASOUND.get_all()
    assert "echonet_dynamic_segment_lv" in TASK_SEG.get_all()


@pytest.mark.parametrize(
    ("channel_mean", "channel_std", "message"),
    [
        ([0.0, 0.0], [1.0] * 3, "three RGB values"),
        ([0.0] * 3, [1.0, 1.0], "three RGB values"),
        ([float("nan"), 0.0, 0.0], [1.0] * 3, "finite values"),
        ([0.0] * 3, [float("inf"), 1.0, 1.0], "finite values"),
        ([True, 0.0, 0.0], [1.0] * 3, "finite values"),
        ([0.0] * 3, [0.0, 1.0, 1.0], "greater than zero"),
        ([0.0] * 3, [-1.0, 1.0, 1.0], "greater than zero"),
    ],
)
def test_invalid_channel_statistics(
    channel_mean: list[float], channel_std: list[float], message: str
) -> None:
    """Reject invalid normalization parameters before model inference."""
    with pytest.raises(ValueError, match=message):
        _validate_channel_statistics(channel_mean, channel_std)


@pytest.mark.slow
@pytest.mark.cuda
def test_echonet_dynamic_predict_ef_with_real_video(
    ultrasound_video: bytes,
    channel_statistics: tuple[list[float], list[float]],
) -> None:
    """Predict a finite EF and report the published temporal sampling."""
    channel_mean, channel_std = channel_statistics
    try:
        result = echonet_dynamic_predict_ef(ultrasound_video, channel_mean, channel_std)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert result["model"] == "EchoNet-Dynamic"
    assert isinstance(result["ef_percent"], float)
    assert math.isfinite(result["ef_percent"])
    assert result["clip_frames"] == 32
    assert result["sampling_period_frames"] == 2


@pytest.mark.slow
@pytest.mark.cuda
def test_echonet_dynamic_segment_lv_with_real_image(
    ultrasound_image: bytes,
    channel_statistics: tuple[list[float], list[float]],
) -> None:
    """Return a binary palette PNG mask at the original frame dimensions."""
    channel_mean, channel_std = channel_statistics
    try:
        result = echonet_dynamic_segment_lv(ultrasound_image, channel_mean, channel_std)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert isinstance(result, bytes)
    with (
        Image.open(BytesIO(ultrasound_image)) as source,
        Image.open(BytesIO(result)) as mask,
    ):
        assert mask.format == "PNG"
        assert mask.mode == "P"
        assert mask.size == source.size
        assert mask.info["transparency"] == 0
        assert set(np.unique(np.asarray(mask))) <= {0, 1}
