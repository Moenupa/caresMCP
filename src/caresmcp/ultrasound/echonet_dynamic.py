"""EchoNet-Dynamic tools for apical four-chamber echocardiography videos."""

from functools import lru_cache
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_SEG, ULTRASOUND
from caresmcp.util.moduleloader import requires
from caresmcp.util.transformers import encode_label_mask

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


def _validate_channel_statistics(
    channel_mean: list[float], channel_std: list[float]
) -> None:
    """Validate RGB pixel statistics used by the published checkpoint."""
    if len(channel_mean) != 3 or len(channel_std) != 3:
        raise ValueError(
            "channel_mean and channel_std must each contain three RGB values"
        )
    statistics = (*channel_mean, *channel_std)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        for value in statistics
    ):
        raise ValueError("channel_mean and channel_std must contain finite values")
    if any(value <= 0 for value in channel_std):
        raise ValueError("channel_std values must be greater than zero")


def _decode_echonet_dynamic_video(
    video: bytes,
    channel_mean: list[float],
    channel_std: list[float],
    cv2: Any,
    torch: Any,
) -> Any:
    """Decode a centered 32-frame clip at a two-frame sampling period."""
    import numpy as np

    if not video:
        raise ValueError("video must contain a readable apical four-chamber echo")
    _validate_channel_statistics(channel_mean, channel_std)

    with TemporaryDirectory() as temporary_directory:
        video_path = Path(temporary_directory) / "input.mp4"
        video_path.write_bytes(video)
        capture = cv2.VideoCapture(video_path.as_posix())
        try:
            if not capture.isOpened():
                raise ValueError(
                    "video must contain a readable apical four-chamber echo"
                )

            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_total <= 0:
                raise ValueError(
                    "video must contain a readable apical four-chamber echo"
                )

            padded_frame_total = max(frame_total, _FRAME_COUNT * _FRAME_PERIOD)
            last_start = padded_frame_total - ((_FRAME_COUNT - 1) * _FRAME_PERIOD + 1)
            start = last_start // 2
            frame_indices = [
                start + _FRAME_PERIOD * index for index in range(_FRAME_COUNT)
            ]
            frames = []
            for frame_index in frame_indices:
                if frame_index >= frame_total:
                    frames.append(
                        np.zeros((_FRAME_HEIGHT, _FRAME_WIDTH, 3), dtype=np.float32)
                    )
                    continue

                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                read_successfully, frame = capture.read()
                if not read_successfully:
                    raise ValueError(
                        "video must contain a readable apical four-chamber echo"
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(
                    rgb,
                    (_FRAME_WIDTH, _FRAME_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
                frames.append(resized.astype(np.float32))
        finally:
            capture.release()

    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    mean = tensor.new_tensor(channel_mean).view(1, 3, 1, 1)
    standard_deviation = tensor.new_tensor(channel_std).view(1, 3, 1, 1)
    tensor = (tensor - mean) / standard_deviation

    # The zero frames above represent padding after normalization, as in the
    # EchoNet-Dynamic dataset loader.
    tensor[torch.from_numpy(np.asarray(frame_indices) >= frame_total)] = 0
    return tensor.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)


@lru_cache(maxsize=1)
def _load_echonet_dynamic_model() -> Any:
    """Load and cache the released EchoNet-Dynamic EF checkpoint."""
    import torch
    from torchvision.models.video import r2plus1d_18

    model = r2plus1d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    checkpoint = torch.hub.load_state_dict_from_url(
        _EF_CHECKPOINT_URL,
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


@lru_cache(maxsize=1)
def _load_echonet_dynamic_segmentation_model() -> Any:
    """Load and cache the released EchoNet-Dynamic LV segmentation checkpoint."""
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
@requires(["torch", "torchvision", "cv2"], extra="ultrasound")
def echonet_dynamic_predict_ef(
    video: bytes,
    channel_mean: list[float],
    channel_std: list[float],
) -> dict[str, Any]:
    """Estimate left ventricular ejection fraction from an apical four-chamber video.

    Args:
        video: Apical four-chamber echocardiography video bytes.
        channel_mean: RGB pixel means computed from EchoNet-Dynamic training
            videos, on the original 0 to 255 pixel scale.
        channel_std: RGB pixel standard deviations computed from
            EchoNet-Dynamic training videos, on the original 0 to 255 pixel
            scale.

    Returns:
        Model name, estimated ejection fraction in percent, and the temporal
        sampling used for the input clip.

    See Also:
        https://github.com/echonet/dynamic
        https://doi.org/10.1038/s41586-020-2145-8
    """
    import cv2
    import torch

    tensor = _decode_echonet_dynamic_video(
        video,
        channel_mean,
        channel_std,
        cv2,
        torch,
    )
    model = _load_echonet_dynamic_model()
    with torch.inference_mode():
        prediction = model(tensor)

    if prediction.ndim != 2 or prediction.shape != (1, 1):
        raise RuntimeError("EchoNet-Dynamic must return one ejection fraction value")

    return {
        "model": "EchoNet-Dynamic",
        "ef_percent": float(prediction[0, 0].detach().float().cpu()),
        "clip_frames": _FRAME_COUNT,
        "sampling_period_frames": _FRAME_PERIOD,
    }


@ULTRASOUND.register("echonet_dynamic_segment_lv")
@TASK_SEG.register("echonet_dynamic_segment_lv")
@requires(["torch", "torchvision", "PIL"], extra="ultrasound")
def echonet_dynamic_segment_lv(
    image: bytes,
    channel_mean: list[float],
    channel_std: list[float],
) -> bytes:
    """Segment the left ventricle in an apical four-chamber echocardiogram frame.

    Args:
        image: Apical four-chamber echocardiography frame bytes.
        channel_mean: RGB pixel means computed from EchoNet-Dynamic training
            videos, on the original 0 to 255 pixel scale.
        channel_std: RGB pixel standard deviations computed from
            EchoNet-Dynamic training videos, on the original 0 to 255 pixel
            scale.

    Returns:
        Palette PNG mask with transparent background and pixel values 0 for
        background and 1 for the predicted left ventricle.

    See Also:
        https://github.com/echonet/dynamic
        https://doi.org/10.1038/s41586-020-2145-8
    """
    from io import BytesIO

    import numpy as np
    import torch
    from PIL import Image, UnidentifiedImageError
    from torch.nn import functional

    _validate_channel_statistics(channel_mean, channel_std)
    try:
        with Image.open(BytesIO(image)) as source:
            original_size = source.size
            resized = source.convert("RGB").resize(
                (_FRAME_WIDTH, _FRAME_HEIGHT), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(resized, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("image must contain a readable echocardiogram frame") from exc

    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    mean = tensor.new_tensor(channel_mean).view(1, 3, 1, 1)
    standard_deviation = tensor.new_tensor(channel_std).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / standard_deviation).to(DEVICE)

    model = _load_echonet_dynamic_segmentation_model()
    with torch.inference_mode():
        logits = model(tensor)["out"]
        logits = functional.interpolate(
            logits,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )
    labels = (logits[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
    return encode_label_mask(labels)
