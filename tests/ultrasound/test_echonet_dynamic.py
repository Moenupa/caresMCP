"""Reproduce EchoNet-Dynamic on the authors' released cardiac examples.

The Nature 2020 report (doi:10.1038/s41586-020-2145-8) claims an ejection
fraction MAE of 4.1% and a left-ventricle Dice similarity coefficient of 0.92
on the Stanford test split, which is part of the 10,030-video dataset behind a
data use agreement; those cohort scores are not reproduced here. The README
releases nine examples and states that three patients have normal cardiac
function, three have low ejection fractions, and three have arrhythmia, while
each video's right panel renders the authors' own segmentation over the
original frame. One example per group checks that the tool recovers the
released masks and follows the released function grouping.
"""

import math

import numpy as np
import pytest

import caresmcp.ultrasound.echonet_dynamic as echonet
from caresmcp.api.decorators import save_returns
from caresmcp.api.io import decode_as
from caresmcp.api.metrics import mask_overlap
from caresmcp.ultrasound.echonet_dynamic import (
    _load_echonet_dynamic_model,
    _load_echonet_dynamic_segmentation_model,
    echonet_dynamic_predict_ef,
    echonet_dynamic_segment_lv,
)

# Released overlay pixels are solid blue; lossy video encoding leaves channel
# differences well above this threshold inside the mask and near zero outside.
RELEASED_OVERLAY_MIN_CHROMA = 60
# Agreement with the released model's rendered masks rather than the 0.92
# human-tracing Dice, which needs the credentialed dataset.
MIN_RELEASED_MASK_DICE = 0.9
# Normal cardiac function is conventionally EF >= 50%; low EF means < 40%.
RELEASED_NORMAL_MIN_EF = 50
RELEASED_LOW_MAX_EF = 40


def _released_overlay_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return the authors' rendered segmentation from the right panel."""
    right = frame_bgr[:, frame_bgr.shape[1] // 2 :]
    chroma = right.max(axis=2).astype(np.int16) - right.min(axis=2).astype(np.int16)
    return chroma > RELEASED_OVERLAY_MIN_CHROMA


def _encode_frame(frame_bgr: np.ndarray) -> bytes:
    """Encode a BGR frame as lossless PNG bytes for the image tool input."""
    import cv2

    encoded, buffer = cv2.imencode(".png", cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    assert encoded
    return buffer.tobytes()


@pytest.mark.slow
@save_returns()
def test_segmentation_matches_released_masks(echonet_example_frames, monkeypatch):
    """Recover the authors' rendered left-ventricle masks."""
    category, frames = echonet_example_frames
    frame_index = len(frames) // 2
    frame = frames[frame_index]
    model = _load_echonet_dynamic_segmentation_model()
    monkeypatch.setattr(
        echonet, "_load_echonet_dynamic_segmentation_model", lambda: model
    )
    reference = _released_overlay_mask(frame)
    assert reference.any(), "released segmentation panel is empty"
    mask = decode_as(echonet_dynamic_segment_lv(_encode_frame(frame)))
    left_width = frame.shape[1] // 2
    overlap = mask_overlap(np.asarray(mask)[:, :left_width], reference)
    assert overlap.dice is not None
    assert overlap.dice > MIN_RELEASED_MASK_DICE
    return {"category": category, "frame_index": frame_index, "dice": overlap.dice}


@pytest.mark.slow
@save_returns()
def test_ef_follows_released_function_groups(echonet_example, monkeypatch):
    """Estimate normal and low ejection fractions as the README groups them."""
    category, video = echonet_example
    model = _load_echonet_dynamic_model()
    monkeypatch.setattr(echonet, "_load_echonet_dynamic_model", lambda: model)
    result = echonet_dynamic_predict_ef(video)
    assert math.isfinite(result["ef_percent"])
    if category == "normal":
        assert result["ef_percent"] >= RELEASED_NORMAL_MIN_EF
    elif category == "low_ef":
        assert result["ef_percent"] < RELEASED_LOW_MAX_EF
    # The arrhythmia example only demonstrates that irregular cycles remain
    # predictable; the README makes no ejection fraction claim for it.
    return result
