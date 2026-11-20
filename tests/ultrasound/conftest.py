"""Official cardiology examples shared by the ultrasound reproduction tests."""

from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from caresmcp.api.decorators import download_bytes

# Revision-pinned checkout of https://github.com/echonet/dynamic. The README
# groups these videos into three normal, three low-ejection-fraction, and three
# arrhythmia patients, and each video's right panel renders the authors' own
# segmentation over the original frame.
ECHONET_REVISION = "e01d02947b20604b0ba1a427f3cf6f4890dc4e9c"
ECHONET_MEDIA_ROOT = (
    f"https://cdn.jsdelivr.net/gh/echonet/dynamic@{ECHONET_REVISION}/docs/media/"
)
# The first released example from each README function group, with the SHA-256
# of the pinned file.
ECHONET_EXAMPLES = {
    "normal": (
        "0X10A28877E97DF540.mp4",
        "d9b6eab31417b6e1bc58ba036c2148cb94e5a378a36df02d3bc9976d85350669",
    ),
    "low_ef": (
        "0X129133A90A61A59D.mp4",
        "d95c109e227b570520d58c8a8a73088a4460be5cadc0e16091d715b6ee605d4f",
    ),
    "arrhythmia": (
        "0X132C1E8DBB715D1D.mp4",
        "382f95d6bea0e9e89da173314beab79356f640488b8ed575669a239ebd0233d4",
    ),
}


@cache
def _example_bytes(category: str) -> bytes:
    """Download one released example once per test session."""
    filename, checksum = ECHONET_EXAMPLES[category]
    return download_bytes(ECHONET_MEDIA_ROOT + filename, checksum=checksum)


def _read_video_frames(video: bytes) -> list[np.ndarray]:
    """Decode video bytes into BGR frames, matching OpenCV's reader layout."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "input.mp4"
        path.write_bytes(video)
        capture = cv2.VideoCapture(path.as_posix())
        if not capture.isOpened():
            raise ValueError("official example video could not be opened")
        try:
            frames = []
            while True:
                read_successfully, frame = capture.read()
                if not read_successfully:
                    break
                frames.append(frame)
        finally:
            capture.release()
    return frames


def _encode_frame(frame_bgr: np.ndarray) -> bytes:
    """Encode a BGR frame as lossless PNG bytes for an image tool input."""
    encoded, buffer = cv2.imencode(".png", cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not encoded:
        raise ValueError("official example frame could not be encoded")
    return buffer.tobytes()


@pytest.fixture(scope="session", params=tuple(ECHONET_EXAMPLES))
def echonet_example(request: pytest.FixtureRequest) -> tuple[str, bytes]:
    """Supply the released video for one README function group."""
    return request.param, _example_bytes(request.param)


@pytest.fixture(scope="session", params=tuple(ECHONET_EXAMPLES))
def echonet_example_frames(
    request: pytest.FixtureRequest,
) -> tuple[str, list[np.ndarray]]:
    """Supply the decoded BGR frames for one README function group."""
    return request.param, _read_video_frames(_example_bytes(request.param))


@pytest.fixture(scope="session")
def echonet_examples() -> dict[str, bytes]:
    """Supply the released video for each README function group."""
    return {category: _example_bytes(category) for category in ECHONET_EXAMPLES}


@pytest.fixture(scope="session")
def echonet_png() -> bytes:
    """Supply the first released normal-example frame as PNG bytes."""
    frame = _read_video_frames(_example_bytes("normal"))[0]
    # The left half is the original frame; the right half is the overlay.
    return _encode_frame(frame[:, : frame.shape[1] // 2])
