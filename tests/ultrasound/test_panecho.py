"""Integration coverage for ultrasound tools using real samples."""

import math
from pathlib import Path
from typing import Any, get_type_hints
from zipfile import ZipFile

import pytest
from pydantic import TypeAdapter, ValidationError

from caresmcp.api.decorators import adapt_byte_inputs, save_returns
from caresmcp.api.models import get_torch_model
from caresmcp.ultrasound.panecho import (
    _decode_panecho_video,
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


@pytest.mark.slow
@save_returns("panecho")
def test_panecho_predict_video(
    video_predictions: dict[str, Any],
):
    """Return finite measurements and valid class probabilities."""
    assert video_predictions["model"] == "PanEcho"
    assert video_predictions["predictions"]
    assert video_predictions["measurement_units"]["GLS"] == "%"
    assert video_predictions["measurement_units"]["RVSP"] == "mmHg"
    assert set(video_predictions["predictions"]["LVWallMotionAbnormalities"]) == {
        "None",
        "Present",
    }
    assert sum(
        video_predictions["predictions"]["LVWallMotionAbnormalities"].values()
    ) == pytest.approx(1)
    for pred in video_predictions["predictions"].values():
        values = pred.values() if isinstance(pred, dict) else (pred,)
        assert all(math.isfinite(value) for value in values)
        if isinstance(pred, dict):
            assert pred
            assert all(0 <= value <= 1 for value in values)
    return video_predictions


@pytest.mark.slow
@pytest.mark.parametrize("video_count", [1, 2])
@save_returns("panecho")
def test_panecho_predict_study(
    video_predictions: dict[str, Any],
    ultrasound_video: bytes,
    video_count: int,
    tmp_path: Path,
):
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
        study_predictions["measurement_units"] == video_predictions["measurement_units"]
    )
    assert (
        study_predictions["predictions"].keys()
        == video_predictions["predictions"].keys()
    )
    for task_name, expected in video_predictions["predictions"].items():
        assert study_predictions["predictions"][task_name] == pytest.approx(expected)
    return study_predictions


def test_panecho_predict_study_requires_videos() -> None:
    """Declare that a study must contain at least one video."""
    videos_type = get_type_hints(panecho_predict_study, include_extras=True)["videos"]
    with pytest.raises(ValidationError):
        TypeAdapter(videos_type).validate_python([])


@pytest.mark.slow
@save_returns("panecho")
def test_panecho_model_output_contract(ultrasound_video: bytes):
    """Check that model outputs match the task metadata used by the tools."""
    import torch

    model: Any = get_torch_model("CarDS-Yale/PanEcho", "PanEcho", tasks="all")
    with torch.inference_mode():
        outputs = model(_decode_panecho_video(ultrasound_video))
    tasks = {task.task_name: task for task in model.tasks}
    assert outputs.keys() == tasks.keys()
    for name, output in outputs.items():
        task = tasks[name]
        values = output.detach().float().cpu().reshape(-1).tolist()
        assert all(math.isfinite(value) for value in values)
        if task.task_type == "regression":
            assert len(values) == 1
        elif task.task_type == "binary_classification":
            assert len(task.class_names) == 2
            assert len(values) == 1
        else:
            assert len(values) == len(task.class_names)
    return {
        name: output.detach().float().cpu().tolist() for name, output in outputs.items()
    }


@pytest.mark.slow
@save_returns("panecho")
def test_panecho_predict_study_with_distinct_views():
    """Aggregate two patient-matched views from the CC BY 4.0 EchoNSTE-MI dataset."""
    fsspec = pytest.importorskip("fsspec")
    pytest.importorskip("aiohttp")
    url = (
        "https://data.mendeley.com/public-files/datasets/y6z64ntstg/"
        "files/afaad34b-4493-4182-a337-2e5449bd2cc3/file_downloaded"
    )
    # HTTP range reads avoid downloading the entire 4.3 GB dataset archive.
    paths = [
        f"EchoNSTE-MI/{view}/NSTEMI/cropped_01_{view}.mp4" for view in ("2CH", "4CH")
    ]
    with (
        fsspec.open(url, "rb", block_size=1024 * 1024) as remote,
        ZipFile(remote) as archive,
    ):
        videos = [archive.read(path) for path in paths]

    per_video = [panecho_predict_video(video) for video in videos]
    study = panecho_predict_study(videos)
    assert study["video_count"] == 2
    assert study["measurement_units"] == per_video[0]["measurement_units"]
    for name, first in per_video[0]["predictions"].items():
        second = per_video[1]["predictions"][name]
        if isinstance(first, dict):
            for label, value in first.items():
                assert study["predictions"][name][label] == pytest.approx(
                    (value + second[label]) / 2
                )
        else:
            assert study["predictions"][name] == pytest.approx((first + second) / 2)
    return study


@pytest.mark.slow
@pytest.mark.parametrize("sample_kind", ["image", "video"])
@save_returns("panecho")
def test_panecho_embedding(sample_kind: str, request: pytest.FixtureRequest):
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
    return result
