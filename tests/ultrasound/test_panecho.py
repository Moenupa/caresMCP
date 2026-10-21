"""Integration coverage for ultrasound tools using real samples."""

import math
from io import BytesIO
from pathlib import Path

import pytest

from caresmcp.registry import TASK_EMBED, TASK_SEG, ULTRASOUND
from caresmcp.ultrasound.echonet_dynamic import (
    echonet_dynamic_predict_ef,
    echonet_dynamic_segment_lv,
)
from caresmcp.ultrasound.panecho import (
    panecho_aggregate_study,
    panecho_image_embed,
    panecho_predict_video,
    panecho_video_embed,
)

SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data_subset10" / "ultrasound"
SAMPLE_VIDEO_PATHS = (
    tuple(
        sorted(
            path
            for path in SAMPLE_DIRECTORY.rglob("*")
            if path.suffix.lower() in {".avi", ".m4v", ".mov", ".mp4"}
        )
    )
    if SAMPLE_DIRECTORY.exists()
    else ()
)
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


def test_panecho_tools_are_registered() -> None:
    """Expose PanEcho inference and embedding tools through the catalogues."""
    assert "panecho_predict_video" in ULTRASOUND.get_all()
    assert "panecho_aggregate_study" in ULTRASOUND.get_all()
    assert "panecho_video_embed" in ULTRASOUND.get_all()
    assert "panecho_image_embed" in ULTRASOUND.get_all()
    assert "panecho_video_embed" in TASK_EMBED.get_all()
    assert "panecho_image_embed" in TASK_EMBED.get_all()


def test_echonet_dynamic_tool_is_registered() -> None:
    """Expose EchoNet-Dynamic EF estimation through the ultrasound catalogue."""
    assert "echonet_dynamic_predict_ef" in ULTRASOUND.get_all()
    assert "echonet_dynamic_segment_lv" in ULTRASOUND.get_all()
    assert "echonet_dynamic_segment_lv" in TASK_SEG.get_all()


@pytest.mark.slow
def test_panecho_with_real_ultrasound_samples() -> None:
    """Run PanEcho on real video and image examples when the subset is present."""
    if not SAMPLE_VIDEO_PATHS or not SAMPLE_IMAGE_PATHS:
        pytest.skip("data_subset10/ultrasound samples are not available")

    video = SAMPLE_VIDEO_PATHS[0].read_bytes()
    image = SAMPLE_IMAGE_PATHS[0].read_bytes()
    try:
        predictions = panecho_predict_video(video)
        study_predictions = panecho_aggregate_study([predictions])
        video_embedding = panecho_video_embed(video)
        image_embedding = panecho_image_embed(image)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert predictions["model"] == "PanEcho"
    assert predictions["predictions"]
    assert study_predictions["video_count"] == 1
    assert study_predictions["aggregation"] == "equal_weight_mean"
    assert study_predictions["predictions"] == predictions["predictions"]
    for prediction in predictions["predictions"].values():
        values = prediction.values() if isinstance(prediction, dict) else (prediction,)
        assert all(math.isfinite(value) for value in values)
    assert len(video_embedding["embedding"]) == 768
    assert len(image_embedding["embedding"]) == 768


@pytest.mark.slow
def test_echonet_dynamic_with_real_ultrasound_video() -> None:
    """Run EchoNet-Dynamic on a real video when an example is available."""
    if not SAMPLE_VIDEO_PATHS:
        pytest.skip("data_subset10/ultrasound video samples are not available")

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(SAMPLE_VIDEO_PATHS[0].as_posix())
    try:
        frames = []
        for _ in range(10):
            read_successfully, frame = capture.read()
            if not read_successfully:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32))
    finally:
        capture.release()
    if not frames:
        pytest.skip("the available ultrasound video could not be decoded")

    pixels = np.stack(frames)
    channel_mean = pixels.mean(axis=(0, 1, 2)).tolist()
    channel_std = pixels.std(axis=(0, 1, 2)).tolist()
    if any(value == 0 for value in channel_std):
        pytest.skip("the available video has a constant color channel")
    try:
        result = echonet_dynamic_predict_ef(
            SAMPLE_VIDEO_PATHS[0].read_bytes(),
            channel_mean=channel_mean,
            channel_std=channel_std,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    assert result["model"] == "EchoNet-Dynamic"
    assert math.isfinite(result["ef_percent"])


@pytest.mark.slow
def test_echonet_dynamic_segmentation_with_real_ultrasound_frame() -> None:
    """Segment a real ultrasound frame when an example is available."""
    if not SAMPLE_IMAGE_PATHS:
        pytest.skip("data_subset10/ultrasound image samples are not available")

    import numpy as np
    from PIL import Image

    image = SAMPLE_IMAGE_PATHS[0].read_bytes()
    with Image.open(BytesIO(image)) as source:
        pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    channel_mean = pixels.mean(axis=(0, 1)).tolist()
    channel_std = pixels.std(axis=(0, 1)).tolist()
    if any(value == 0 for value in channel_std):
        pytest.skip("the available image has a constant color channel")

    try:
        segmentation = echonet_dynamic_segment_lv(
            image,
            channel_mean=channel_mean,
            channel_std=channel_std,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    with Image.open(BytesIO(segmentation)) as mask:
        labels = np.asarray(mask)
        assert mask.size == pixels.shape[1::-1]
        assert set(np.unique(labels)).issubset({0, 1})
