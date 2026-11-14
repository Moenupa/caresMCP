"""EchoNet-Dynamic tools for apical four-chamber echocardiography videos."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, cast

import cv2
import numpy as np
from pydantic import Field

from caresmcp.api.decorators import adapt_inputs, requires
from caresmcp.api.io import decode_as, encode_label_mask
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_REG, TASK_SEG, ULTRASOUND

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

_EF_CHECKPOINT_URL = (
    "https://github.com/echonet/dynamic/releases/download/v1.0.0/"
    "r2plus1d_18_32_2_pretrained.pt"
)
_SEGMENTATION_CHECKPOINT_URL = (
    "https://github.com/echonet/dynamic/releases/download/v1.0.0/"
    "deeplabv3_resnet50_random.pt"
)
_FRAME_COUNT = 32
_FRAME_PERIOD = 2
_FRAME_HEIGHT = 112
_FRAME_WIDTH = 112
_INFERENCE_BATCH_SIZE = 4
# Averaging 32 evenly spaced clips tracks the all-clips mean closely while
# keeping inference bounded for arbitrarily long videos.
_MAX_AVERAGED_CLIPS = 32

# Echocardiography viewers often place the original frame beside an annotated
# or color-overlaid copy.
_PANEL_ASPECT_RATIO = 2.0
_PANEL_ASPECT_TOLERANCE = 0.2
_PANEL_MIN_FOREGROUND = 0.02
_PANEL_CHROMA_RATIO = 1.5
_PANEL_CHROMA_MARGIN = 4.0


class _EchoFrame(NamedTuple):
    """Frame tensor, original dimensions, and source-to-model transform."""

    pixels: "Tensor"
    original_size: tuple[int, int]
    transform: np.ndarray


def _channel_statistics(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = pixels.reshape(-1, pixels.shape[-1])
    mean = flattened.mean(axis=0, dtype=np.float64)
    standard_deviation = flattened.std(axis=0, dtype=np.float64)
    if (standard_deviation <= 0).any():
        raise ValueError("input frames must contain non-constant pixel values")
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def _decode_echonet_dynamic_image(image: bytes) -> _EchoFrame:
    import torch

    source = decode_as(image, mode="RGB")
    pixels = np.asarray(source)
    transform = _echo_transform(pixels)
    resized = _transform_echo_frame(pixels, transform)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    return _EchoFrame(tensor, source.size, transform)


def _select_echo_panel(
    frame: np.ndarray, mask: np.ndarray
) -> tuple[int, int, int, int]:
    """Return the crop box of the original-image panel in a two-panel composite.

    When a frame contains two similarly sized panels and only one is markedly
    colorful, the less colorful panel is treated as the original image rather
    than an annotated copy. Returns the full frame when no such layout is
    detected.
    """
    height, width = mask.shape
    first_box = (0, 0, width // 2, height)
    second_box = (width // 2, 0, width, height)
    if abs(width / height - _PANEL_ASPECT_RATIO) <= _PANEL_ASPECT_TOLERANCE:
        first = frame[:, : width // 2]
        second = frame[:, width // 2 :]
        first_mask = mask[:, : width // 2]
        second_mask = mask[:, width // 2 :]
    elif abs(height / width - _PANEL_ASPECT_RATIO) <= _PANEL_ASPECT_TOLERANCE:
        first_box = (0, 0, width, height // 2)
        second_box = (0, height // 2, width, height)
        first = frame[: height // 2]
        second = frame[height // 2 :]
        first_mask = mask[: height // 2]
        second_mask = mask[height // 2 :]
    else:
        return 0, 0, width, height

    if min(first_mask.mean(), second_mask.mean()) < _PANEL_MIN_FOREGROUND:
        return 0, 0, width, height

    def chroma(panel: np.ndarray) -> float:
        pixels = panel.astype(np.int16)
        return float((pixels.max(axis=2) - pixels.min(axis=2)).mean())

    first_chroma, second_chroma = chroma(first), chroma(second)
    colorful = max(first_chroma, second_chroma)
    plain = min(first_chroma, second_chroma)
    if plain * _PANEL_CHROMA_RATIO >= colorful:
        return 0, 0, width, height
    if colorful - plain < _PANEL_CHROMA_MARGIN:
        return 0, 0, width, height
    return first_box if first_chroma <= second_chroma else second_box


def _echo_transform(frame: np.ndarray) -> np.ndarray:
    """Locate the main image region and fit it into a square without distortion.

    Native-size square inputs are retained as-is, as EchoNet's distributed
    videos already contain a masked sector. For other inputs, locate the largest
    connected region of image content, allowing a margin around the sector.
    Two-panel composites use their less colorful panel, which is usually the
    original frame rather than an annotated copy. This is a heuristic, not a
    view or anatomy detector.
    """
    height, width = frame.shape[:2]
    if height == width and width == _FRAME_WIDTH:
        x, y, side = 0.0, 0.0, float(width)
    else:
        scale = min(1.0, 512 / max(height, width))
        small = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), max(1.0, min(gray.shape) / 100))
        mask = (gray > 8).astype(np.uint8)
        kernel_size = max(3, int(min(gray.shape) * 0.04) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        panel_left, panel_top, panel_right, panel_bottom = _select_echo_panel(
            small, mask
        )
        panel = mask[panel_top:panel_bottom, panel_left:panel_right]
        panel_height, panel_width = panel.shape
        count, _, stats, _ = cv2.connectedComponentsWithStats(panel)
        if count > 1:
            region = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
        else:
            region = np.zeros(5, dtype=int)
        if region[cv2.CC_STAT_AREA] >= 0.03 * panel.size:
            region_width = int(region[cv2.CC_STAT_WIDTH])
            region_height = int(region[cv2.CC_STAT_HEIGHT])
            if (
                panel_width == panel_height
                and max(region_width, region_height) >= 0.85 * panel_width
            ):
                x = panel_left / scale
                y = panel_top / scale
                side = panel_width / scale
            else:
                left = (region[cv2.CC_STAT_LEFT] + panel_left) / scale
                top = (region[cv2.CC_STAT_TOP] + panel_top) / scale
                region_width /= scale
                region_height /= scale
                side = max(region_width, region_height) * 1.1
                x = left + region_width / 2 - side / 2
                y = top + region_height / 2 - side / 2
        else:
            side = max(panel_width, panel_height) / scale
            x = (panel_left + panel_width / 2) / scale - side / 2
            y = (panel_top + panel_height / 2) / scale - side / 2
    factor = _FRAME_WIDTH / side
    return np.array(
        [[factor, 0, -x * factor], [0, factor, -y * factor]], dtype=np.float32
    )


def _transform_echo_frame(frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        frame, transform, (_FRAME_WIDTH, _FRAME_HEIGHT), flags=cv2.INTER_LINEAR
    )


def _decode_echonet_dynamic_video(video: bytes) -> np.ndarray:
    """Decode the full video as resized RGB frames on the 0–255 scale."""
    if not video:
        raise ValueError("video must contain a readable apical four-chamber echo")
    with TemporaryDirectory() as temporary_directory:
        video_path = Path(temporary_directory) / "input.mp4"
        video_path.write_bytes(video)
        capture = cv2.VideoCapture(video_path.as_posix())
        try:
            if not capture.isOpened():
                raise ValueError(
                    "video must contain a readable apical four-chamber echo"
                )
            # A cine loop may begin with a blank or transitional frame. Select
            # one of the first frames to determine a single crop for the video.
            initial_frames = []
            for _ in range(8):
                read_successfully, frame = capture.read()
                if not read_successfully:
                    break
                initial_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not initial_frames:
                raise ValueError(
                    "video must contain a readable apical four-chamber echo"
                )
            representative = max(
                initial_frames,
                key=lambda frame: cv2.resize(frame, (64, 64)).std(),
            )
            transform = _echo_transform(representative)
            frames = [
                _transform_echo_frame(frame, transform) for frame in initial_frames
            ]
            while True:
                read_successfully, frame = capture.read()
                if not read_successfully:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(_transform_echo_frame(rgb, transform))
        finally:
            capture.release()
    minimum_frames = (_FRAME_COUNT - 1) * _FRAME_PERIOD + 1
    if len(frames) < minimum_frames:
        raise ValueError(
            f"video must contain at least {minimum_frames} readable frames"
        )
    return np.stack(frames)


def _clip_starts(frame_count: int) -> np.ndarray:
    """Return evenly spaced clip starts bounded by the averaging budget."""
    last_start = frame_count - ((_FRAME_COUNT - 1) * _FRAME_PERIOD + 1)
    if last_start < _MAX_AVERAGED_CLIPS:
        return np.arange(last_start + 1)
    return np.unique(
        np.linspace(0, last_start, _MAX_AVERAGED_CLIPS).round().astype(int)
    )


def _load_echonet_dynamic_model() -> "Module":
    """Return the pretrained EchoNet-Dynamic ejection fraction model."""
    import torch
    from torchvision.models.video import r2plus1d_18

    model = r2plus1d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    # The released checkpoint stores NumPy scalars, which PyTorch's safe
    # unpickler rejects. It is downloaded over HTTPS from the official release.
    checkpoint = torch.hub.load_state_dict_from_url(
        _EF_CHECKPOINT_URL,
        map_location="cpu",
        progress=True,
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        name.removeprefix("module."): value for name, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def _load_echonet_dynamic_segmentation_model() -> "Module":
    """Return the pretrained EchoNet-Dynamic left ventricle segmentation model."""
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=False,
    )
    final_layer = model.classifier[-1]
    model.classifier[-1] = torch.nn.Conv2d(
        final_layer.in_channels,
        1,
        kernel_size=final_layer.kernel_size,
        stride=final_layer.stride,
        padding=final_layer.padding,
        dilation=final_layer.dilation,
        groups=final_layer.groups,
        bias=final_layer.bias is not None,
        padding_mode=final_layer.padding_mode,
    )
    checkpoint = torch.hub.load_state_dict_from_url(
        _SEGMENTATION_CHECKPOINT_URL,
        map_location="cpu",
        progress=True,
        weights_only=True,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        name.removeprefix("module."): value for name, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


@ULTRASOUND.register("echonet_dynamic_predict_ef")
@TASK_REG.register("echonet_dynamic_predict_ef")
@requires(["torch", "torchvision"], extra="ultrasound")
@adapt_inputs(video=_decode_echonet_dynamic_video)
def echonet_dynamic_predict_ef(
    video: Annotated[
        bytes,
        Field(
            description="Apical four-chamber echocardiography video with at least 63 readable frames."
        ),
    ],
) -> dict[str, Any]:
    """Estimate left ventricular ejection fraction from an apical four-chamber video.

    Up to 32 clips are sampled evenly across the video and averaged. Normalization
    uses per-video channel statistics because EchoNet-Dynamic's training statistics
    were not published.

    See Also:
        https://github.com/echonet/dynamic
        https://doi.org/10.1038/s41586-020-2145-8
    """
    import torch

    pixels = cast(np.ndarray, video)
    channel_mean, channel_std = _channel_statistics(pixels)
    starts = _clip_starts(len(pixels))
    model = _load_echonet_dynamic_model()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(starts), _INFERENCE_BATCH_SIZE):
            clips = np.stack(
                [
                    pixels[start : start + _FRAME_COUNT * _FRAME_PERIOD : _FRAME_PERIOD]
                    for start in starts[offset : offset + _INFERENCE_BATCH_SIZE]
                ]
            ).astype(np.float32)
            clips = (clips - channel_mean) / channel_std
            tensor = torch.from_numpy(clips).permute(0, 4, 1, 2, 3).to(DEVICE)
            predictions.extend(model(tensor)[:, 0].float().cpu().tolist())
    if not np.isfinite(predictions).all():
        raise RuntimeError("EchoNet-Dynamic returned a non-finite ejection fraction")
    last_start = len(pixels) - ((_FRAME_COUNT - 1) * _FRAME_PERIOD + 1)
    return {
        "model": "EchoNet-Dynamic",
        "ef_percent": float(np.mean(predictions)),
        "clip_ef_range_percent": [min(predictions), max(predictions)],
        "clips_used": len(predictions),
        "clips_available": int(last_start + 1),
        "normalization": "per-video channel statistics (training statistics unpublished)",
        "clip_frames": _FRAME_COUNT,
        "sampling_period_frames": _FRAME_PERIOD,
    }


@ULTRASOUND.register("echonet_dynamic_segment_lv")
@TASK_SEG.register("echonet_dynamic_segment_lv")
@requires(["torch", "torchvision"], extra="ultrasound")
@adapt_inputs(image=_decode_echonet_dynamic_image)
def echonet_dynamic_segment_lv(
    image: Annotated[
        bytes,
        Field(description="Apical four-chamber echocardiography frame bytes."),
    ],
) -> bytes:
    """Segment the left ventricle in an apical four-chamber echocardiogram frame.

    Normalization uses per-frame channel statistics because EchoNet-Dynamic's
    training statistics were not published.

    Returns:
        PNG mask at the input image size: 0 is transparent background and
        1 (blue) is the predicted left ventricle.

    See Also:
        https://github.com/echonet/dynamic
        https://doi.org/10.1038/s41586-020-2145-8
    """
    import torch

    frame = cast(_EchoFrame, image)

    pixels = frame.pixels[0].permute(1, 2, 0).numpy()
    channel_mean, channel_std = _channel_statistics(pixels)
    mean_tensor = torch.as_tensor(channel_mean).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(channel_std).view(1, 3, 1, 1)
    tensor = ((frame.pixels - mean_tensor) / std_tensor).to(DEVICE)
    model = _load_echonet_dynamic_segmentation_model()
    with torch.inference_mode():
        logits = model(tensor)["out"]
    if not torch.isfinite(logits).all():
        raise RuntimeError("EchoNet-Dynamic returned non-finite segmentation logits")
    logits_array = logits[0, 0].detach().cpu().numpy()
    source_logits = cv2.warpAffine(
        logits_array,
        cv2.invertAffineTransform(frame.transform),
        frame.original_size,
        flags=cv2.INTER_LINEAR,
        borderValue=-1,
    )
    labels = (source_logits > 0).astype(np.uint8)
    return encode_label_mask(labels)
