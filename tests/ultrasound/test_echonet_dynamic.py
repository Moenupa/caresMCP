"""EchoNet-Dynamic coverage using the authors' cardiac example."""

import math
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

import caresmcp.ultrasound.echonet_dynamic as echonet
from caresmcp.api.decorators import save_returns
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.ultrasound.echonet_dynamic import (
    _load_echonet_dynamic_model,
    _load_echonet_dynamic_segmentation_model,
    echonet_dynamic_predict_ef,
    echonet_dynamic_segment_lv,
)

_CARDIAC_EXAMPLE = (
    "https://raw.githubusercontent.com/echonet/dynamic/master/"
    "docs/media/0X10A28877E97DF540.gif"
)


@pytest.fixture(scope="session")
def cardiac_gif():
    """Return EchoNet's real example with original and overlay panels."""
    return _read_bytes(_CARDIAC_EXAMPLE)


@pytest.fixture(scope="session")
def cardiac_frames(cardiac_gif):
    """Use the unannotated left panel of EchoNet's real example."""
    with Image.open(BytesIO(cardiac_gif)) as video:
        frames = []
        for index in range(getattr(video, "n_frames", 1)):
            video.seek(index)
            frames.append(np.asarray(video.convert("RGB").crop((0, 0, 112, 112))))
    return frames


@pytest.fixture
def cardiac_video(cardiac_frames, tmp_path):
    def make_video(count, letterboxed=False):
        path = tmp_path / f"{count}_{letterboxed}.avi"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (224 if letterboxed else 112, 112),
        )
        if not writer.isOpened():
            pytest.fail("could not create example video")
        try:
            for frame in cardiac_frames[:count]:
                if letterboxed:
                    frame = cv2.copyMakeBorder(frame, 0, 0, 56, 56, cv2.BORDER_CONSTANT)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return path.read_bytes()

    return make_video


@pytest.mark.slow
@save_returns("echonet_dynamic", unpack=True)
def test_echonet_dynamic_predict_ef_with_real_video(cardiac_video, monkeypatch):
    """Average every valid clip in a real cardiac video."""
    import torch

    outputs = []
    model = _load_echonet_dynamic_model()
    monkeypatch.setattr(echonet, "_load_echonet_dynamic_model", lambda: model)
    hook = model.register_forward_hook(
        lambda _model, _inputs, output: outputs.append(output)
    )
    try:
        result = echonet_dynamic_predict_ef(cardiac_video(80))
    finally:
        hook.remove()

    clip_predictions = torch.cat(outputs).flatten().cpu().numpy()
    assert len(clip_predictions) == result["clips_used"] == 18
    assert np.isfinite(clip_predictions).all()
    assert result["ef_percent"] == pytest.approx(np.mean(clip_predictions))
    assert result["clip_ef_range_percent"] == pytest.approx(
        [min(clip_predictions), max(clip_predictions)]
    )
    assert result["clip_frames"] == 32
    assert result["sampling_period_frames"] == 2
    letterboxed = echonet_dynamic_predict_ef(cardiac_video(80, letterboxed=True))
    assert letterboxed["ef_percent"] == pytest.approx(result["ef_percent"], abs=5)
    return {"original": result, "letterboxed": letterboxed}


@pytest.mark.slow
@save_returns("echonet_dynamic", unpack=True)
def test_echonet_dynamic_segment_lv_with_real_image(
    cardiac_gif, cardiac_frames, monkeypatch
):
    """Recover a similar mask when the real frame is enlarged and letterboxed."""
    model = _load_echonet_dynamic_segmentation_model()
    monkeypatch.setattr(
        echonet, "_load_echonet_dynamic_segmentation_model", lambda: model
    )
    source = Image.fromarray(cardiac_frames[0])
    enlarged = source.resize((224, 224))
    letterboxed = ImageOps.expand(enlarged, border=(112, 0, 112, 0), fill=0)
    square_padded = ImageOps.expand(enlarged, border=112, fill=0)

    def segment(image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        result = echonet_dynamic_segment_lv(buffer.getvalue())
        return result, Image.open(BytesIO(result))

    original_bytes, original_mask = segment(source)
    letterbox_bytes, letterbox_mask = segment(letterboxed)
    square_bytes, square_mask = segment(square_padded)
    assert original_mask.size == source.size
    assert letterbox_mask.size == letterboxed.size
    assert square_mask.size == square_padded.size
    original = np.asarray(original_mask) > 0
    restored = (
        np.asarray(
            letterbox_mask.crop((112, 0, 336, 224)).resize(
                (112, 112), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    assert original.any() and restored.any()
    dice = (
        2 * np.logical_and(original, restored).sum() / (original.sum() + restored.sum())
    )
    assert dice > 0.6
    square_restored = (
        np.asarray(
            square_mask.crop((112, 112, 336, 336)).resize(
                (112, 112), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    square_dice = (
        2
        * np.logical_and(original, square_restored).sum()
        / (original.sum() + square_restored.sum())
    )
    assert square_dice > 0.6
    with Image.open(BytesIO(cardiac_gif)) as video:
        composite = Image.fromarray(np.asarray(video.convert("RGB")))
    composite_bytes, composite_mask = segment(composite)
    assert composite_mask.size == composite.size
    composite_left = np.asarray(composite_mask.crop((0, 0, 112, 112))) > 0
    composite_dice = (
        2
        * np.logical_and(original, composite_left).sum()
        / (original.sum() + composite_left.sum())
    )
    assert composite_dice > 0.9
    return {
        "original_mask": original_bytes,
        "letterboxed_mask": letterbox_bytes,
        "square_padded_mask": square_bytes,
        "composite_mask": composite_bytes,
    }


@pytest.mark.slow
@save_returns("echonet_dynamic")
def test_echonet_dynamic_short_video_boundaries(cardiac_video):
    """Reject incomplete clips and accept the minimum complete clip."""
    for count in (1, 62):
        with pytest.raises(ValueError, match="at least 63 readable frames"):
            echonet_dynamic_predict_ef(cardiac_video(count))
    result = echonet_dynamic_predict_ef(cardiac_video(63))
    assert result["clips_used"] == 1
    assert result["clips_available"] == 1
    assert math.isfinite(result["ef_percent"])
    return result


@pytest.mark.slow
@save_returns("echonet_dynamic")
def test_echonet_dynamic_long_video_is_capped(cardiac_frames, cardiac_video):
    """Bound long-video averaging to 32 evenly spaced clips."""
    result = echonet_dynamic_predict_ef(cardiac_video(len(cardiac_frames)))
    assert result["clips_used"] == 32
    assert result["clips_available"] > 32
    assert math.isfinite(result["ef_percent"])
    return result
