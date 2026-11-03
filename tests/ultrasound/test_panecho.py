"""Integration coverage for ultrasound tools using real samples."""

import math
from pathlib import Path

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import TASK_EMBED, ULTRASOUND
from caresmcp.ultrasound.panecho import (
    panecho_aggregate_study,
    panecho_image_embed,
    panecho_predict_video,
    panecho_video_embed,
)

SAMPLES = """
https://prod-images-static.radiopaedia.org/images/75936103/66061f99ceed45ca01fd96c37182847f60d0f915e419b06ed5492510e4b6a0d8_big_gallery.jpeg
https://prod-images-static.radiopaedia.org/images/75936110/94f539f7b8fef0a1695c4c25f03072f6768c8a580c03927f11121d127c7fe8a0.mp4
""".strip().splitlines()


def test_panecho_tools_are_registered() -> None:
    """Expose PanEcho inference and embedding tools through the catalogues."""
    assert "panecho_predict_video" in ULTRASOUND.get_all()
    assert "panecho_aggregate_study" in ULTRASOUND.get_all()
    assert "panecho_video_embed" in ULTRASOUND.get_all()
    assert "panecho_image_embed" in ULTRASOUND.get_all()
    assert "panecho_video_embed" in TASK_EMBED.get_all()
    assert "panecho_image_embed" in TASK_EMBED.get_all()


@pytest.mark.slow
@pytest.mark.cuda
def test_panecho_with_real_ultrasound_samples(
    video_url: str = SAMPLES[1], image_url: str = SAMPLES[0]
) -> None:
    """Run PanEcho on real video and image examples when the subset is present."""
    video = _read_bytes(video_url)
    image = _read_bytes(image_url)
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
    save_results(
        predictions, CACHE_ROOT / "panecho_predict_video" / Path(video_url).parent.stem
    )
    save_results(
        study_predictions,
        CACHE_ROOT / "panecho_aggregate_study" / Path(video_url).parent.stem,
    )
