"""Integration coverage for ultrasound tools using real samples."""

import math
from pathlib import Path
from typing import Any

import pytest

from caresmcp.api.decorators import adapt_byte_inputs
from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import TASK_EMBED, ULTRASOUND
from caresmcp.ultrasound.panecho import (
    panecho_image_embed,
    panecho_predict_study,
    panecho_predict_video,
    panecho_video_embed,
)


@pytest.fixture(scope="session")
def video_predictions(ultrasound_video: bytes) -> dict[str, Any]:
    """Reuse predictions from the real video across inference and study tests."""
    try:
        return panecho_predict_video(ultrasound_video)
    except ImportError as exc:
        pytest.skip(str(exc))


def test_panecho_tools_are_registered() -> None:
    """Expose PanEcho inference and embedding tools through the catalogues."""
    assert "panecho_predict_video" in ULTRASOUND.get_all()
    assert "panecho_predict_study" in ULTRASOUND.get_all()
    assert "panecho_video_embed" in ULTRASOUND.get_all()
    assert "panecho_image_embed" in ULTRASOUND.get_all()
    assert "panecho_video_embed" in TASK_EMBED.get_all()
    assert "panecho_image_embed" in TASK_EMBED.get_all()


@pytest.mark.slow
@pytest.mark.cuda
def test_panecho_predict_video(
    video_predictions: dict[str, Any],
) -> None:
    """Return finite measurements and valid class probabilities."""
    assert video_predictions["model"] == "PanEcho"
    assert video_predictions["predictions"]
    for pred in video_predictions["predictions"].values():
        values = pred.values() if isinstance(pred, dict) else (pred,)
        assert all(math.isfinite(value) for value in values)
        if isinstance(pred, dict):
            assert pred
            assert all(0 <= value <= 1 for value in values)
    save_results(video_predictions, CACHE_ROOT / "panecho_predict_video")


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize("video_count", [1, 2])
def test_panecho_predict_study(
    video_predictions: dict[str, Any],
    ultrasound_video: bytes,
    video_count: int,
    tmp_path: Path,
) -> None:
    """Predict studies from real video bytes and adapted file inputs."""
    if video_count == 1:
        study_predictions = panecho_predict_study([ultrasound_video])
    else:
        video_path = tmp_path / "echo.mp4"
        video_path.write_bytes(ultrasound_video)
        predict_study: Any = adapt_byte_inputs(panecho_predict_study)
        study_predictions = predict_study([video_path] * video_count)
    assert study_predictions["model"] == "PanEcho"
    assert study_predictions["video_count"] == video_count
    assert study_predictions["aggregation"] == "equal_weight_mean"
    assert (
        study_predictions["predictions"].keys()
        == video_predictions["predictions"].keys()
    )
    for task_name, expected in video_predictions["predictions"].items():
        assert study_predictions["predictions"][task_name] == pytest.approx(expected)
    save_results(study_predictions, CACHE_ROOT / "panecho_predict_study")


def test_panecho_predict_study_requires_videos() -> None:
    """Reject a study without videos before running inference."""
    with pytest.raises(ValueError, match="at least one"):
        panecho_predict_study([])


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize("sample_kind", ["image", "video"])
def test_panecho_embedding(sample_kind: str, request: pytest.FixtureRequest) -> None:
    """Return finite 768-dimensional image and video embeddings."""
    sample = request.getfixturevalue(f"ultrasound_{sample_kind}")
    embed = panecho_image_embed if sample_kind == "image" else panecho_video_embed
    try:
        result = embed(sample)
    except ImportError as exc:
        pytest.skip(str(exc))
    assert result["model"] == "PanEcho"
    assert len(result["embedding"]) == 768
    assert all(math.isfinite(value) for value in result["embedding"])
