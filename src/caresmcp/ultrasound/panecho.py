"""Echocardiography tools powered by the pretrained PanEcho model."""

from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Annotated, Any

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as
from caresmcp.api.models import get_torch_model
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_CLS, TASK_EMB, TASK_REG, ULTRASOUND

if TYPE_CHECKING:
    from torch import Tensor
    from torchvision.transforms import Compose

_PANECHO_REPOSITORY = "CarDS-Yale/PanEcho"
_PANECHO_NAME = "PanEcho"
_FRAME_COUNT = 16
_IMAGE_SIZE = 224
_PANECHO_DEPS = ["torch", "torchvision", "timm"]
_MEASUREMENT_UNITS = {
    "EF": "%",
    "GLS": "%",
    "LVEDV": "cm^3",
    "LVESV": "cm^3",
    "LVSV": "cm^3",
    "IVSd": "cm",
    "LVPWd": "cm",
    "LVIDs": "cm",
    "LVIDd": "cm",
    "LVOTDiam": "cm",
    "E|EAvg": "dimensionless",
    "RVSP": "mmHg",
    "RVIDd": "cm",
    "TAPSE": "cm",
    "RVSVel": "cm/s",
    "LAIDs2D": "cm",
    "LAVol": "cm^3",
    "RADimensionM-L(cm)": "cm",
    "AVPkVel(m|s)": "m/s",
    "TVPkGrad": "mmHg",
    "AORoot": "cm",
}
# Upstream flips these binary heads so their scalar denotes class_names[0].
_BINARY_FIRST_CLASS = frozenset(
    {
        "MVStenosis",
        "AVStructure",
        "RASize",
        "RVSystolicFunction",
        "LVWallThickness-increased-modsev",
        "LVWallThickness-increased-any",
        "pericardial-effusion",
    }
)


def _panecho_transform() -> "Compose":
    """Return the PanEcho input transform for RGB echo frames."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (_IMAGE_SIZE, _IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _decode_panecho_video(video: bytes) -> Any:
    """Return a PanEcho input tensor for an encoded echo video."""
    import torch

    with TemporaryDirectory() as temporary_directory:
        video_path = Path(temporary_directory) / "input.mp4"
        video_path.write_bytes(video)
        capture = cv2.VideoCapture(video_path.as_posix())
        try:
            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            if frame_total > 0:
                frame_indices = np.linspace(
                    0, frame_total - 1, num=_FRAME_COUNT, dtype=int
                )
                for frame_index in frame_indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    _, frame = capture.read()
                    frames.append(frame)
            else:
                while True:
                    read_successfully, frame = capture.read()
                    if not read_successfully:
                        break
                    frames.append(frame)
                frame_indices = np.linspace(
                    0, len(frames) - 1, num=_FRAME_COUNT, dtype=int
                )
                frames = [frames[int(index)] for index in frame_indices]
        finally:
            capture.release()

    transform = _panecho_transform()
    tensor = torch.stack(
        [
            transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            for frame in frames
        ]
    )
    return tensor.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)


def _decode_panecho_image(image: bytes) -> "Tensor":
    """Return a PanEcho input tensor for an encoded echo frame."""
    transform = _panecho_transform()
    tensor = transform(decode_as(image, mode="RGB"))
    return tensor.unsqueeze(0).to(DEVICE)


def _serialize_panecho_predictions(model: Any, outputs: Any) -> dict[str, Any]:
    """Return task measurements and class probabilities keyed by task name."""
    task_by_name = {task.task_name: task for task in model.tasks}
    predictions: dict[str, Any] = {}
    for task_name, output in outputs.items():
        task = task_by_name[task_name]

        values = output.detach().float().cpu().reshape(-1).tolist()
        if task.task_type == "regression":
            predictions[task_name] = float(
                -values[0] if task_name == "GLS" else values[0]
            )
            continue

        class_names = [str(label) for label in task.class_names]
        if task.task_type == "binary_classification":
            first = values[0] if task_name in _BINARY_FIRST_CLASS else 1 - values[0]
            predictions[task_name] = {
                class_names[0]: float(first),
                class_names[1]: float(1 - first),
            }
            continue
        predictions[task_name] = {
            label: float(value) for label, value in zip(class_names, values)
        }
    return predictions


@ULTRASOUND.register("panecho_predict_video")
@TASK_CLS.register("panecho_predict_video")
@TASK_REG.register("panecho_predict_video")
@requires(_PANECHO_DEPS, extra="ultrasound")
def panecho_predict_video(
    video: Annotated[
        bytes,
        Field(
            description="Echocardiography cine video, such as MP4. Sixteen frames are sampled uniformly across the video. DICOM is not supported."
        ),
    ],
) -> dict[str, Any]:
    """Predict echocardiography measurements and findings from one video.

    Returns:
        Predictions keyed by task name and units for numeric measurements.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
        https://doi.org/10.1001/jama.2025.8731
    """
    import torch

    tensor = _decode_panecho_video(video)
    model = get_torch_model(
        _PANECHO_REPOSITORY,
        _PANECHO_NAME,
        tasks="all",
    )
    with torch.inference_mode():
        outputs = model(tensor)

    return {
        "model": "PanEcho",
        "predictions": _serialize_panecho_predictions(model, outputs),
        "measurement_units": {
            name: _MEASUREMENT_UNITS[name]
            for name in outputs
            if name in _MEASUREMENT_UNITS
        },
    }


@ULTRASOUND.register("panecho_predict_study")
@TASK_CLS.register("panecho_predict_study")
@TASK_REG.register("panecho_predict_study")
@requires(_PANECHO_DEPS, extra="ultrasound")
def panecho_predict_study(
    videos: Annotated[
        list[bytes],
        Field(
            min_length=1,
            description="Echocardiography cine videos from one study, such as MP4. Sixteen frames are sampled uniformly from each video. DICOM is not supported.",
        ),
    ],
) -> dict[str, Any]:
    """Predict echocardiography measurements and findings for one study.

    Returns:
        Study predictions and measurement units, with equal weight per video.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
        https://doi.org/10.1001/jama.2025.8731
    """
    video_results = [panecho_predict_video(video) for video in videos]
    predictions = [result["predictions"] for result in video_results]
    aggregated: dict[str, Any] = {}
    for task_name, first_value in predictions[0].items():
        if isinstance(first_value, dict):
            aggregated[task_name] = {
                label: fmean(prediction[task_name][label] for prediction in predictions)
                for label in first_value
            }
        else:
            aggregated[task_name] = fmean(
                prediction[task_name] for prediction in predictions
            )

    return {
        "model": "PanEcho",
        "video_count": len(videos),
        "aggregation": "equal_weight_mean",
        "predictions": aggregated,
        "measurement_units": video_results[0]["measurement_units"],
    }


@ULTRASOUND.register("panecho_video_embed")
@TASK_EMB.register("panecho_video_embed")
@requires(_PANECHO_DEPS, extra="ultrasound")
def panecho_video_embed(
    video: Annotated[
        bytes,
        Field(
            description="Echocardiography cine video, such as MP4. Sixteen frames are sampled uniformly across the video. DICOM is not supported."
        ),
    ],
) -> dict[str, Any]:
    """Extract a 768-dimensional PanEcho feature vector from an echo video.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
    """
    import torch

    tensor = _decode_panecho_video(video)
    model = get_torch_model(
        _PANECHO_REPOSITORY,
        _PANECHO_NAME,
        backbone_only=True,
    )
    with torch.inference_mode():
        embedding = model(tensor)[0].detach().float().cpu().tolist()

    return {"model": "PanEcho", "embedding": [float(value) for value in embedding]}


@ULTRASOUND.register("panecho_image_embed")
@TASK_EMB.register("panecho_image_embed")
@requires(_PANECHO_DEPS, extra="ultrasound")
def panecho_image_embed(
    image: Annotated[
        bytes,
        Field(description="Single encoded echocardiography frame."),
    ],
) -> dict[str, Any]:
    """Extract a 768-dimensional PanEcho feature vector from one echo frame.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
    """
    import torch

    tensor = _decode_panecho_image(image)
    model = get_torch_model(
        _PANECHO_REPOSITORY,
        _PANECHO_NAME,
        image_encoder_only=True,
    )
    with torch.inference_mode():
        embedding = model(tensor)[0].detach().float().cpu().tolist()

    return {"model": "PanEcho", "embedding": [float(value) for value in embedding]}
