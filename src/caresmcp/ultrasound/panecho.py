"""Echocardiography tools powered by the pretrained PanEcho model."""

from functools import lru_cache
from math import isfinite
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any

from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_EMBED, ULTRASOUND
from caresmcp.util.moduleloader import requires

_PANECHO_REPOSITORY = "CarDS-Yale/PanEcho"
_FRAME_COUNT = 16
_IMAGE_SIZE = 224
_PANECHO_DEPS = ["torch", "torchvision", "cv2", "timm", "PIL"]


def _decode_panecho_video(video: bytes, cv2: Any, torch: Any) -> Any:
    """Decode and normalize 16 uniformly sampled frames from an echo video."""
    import numpy as np

    if not video:
        raise ValueError("video must contain a readable echocardiogram")

    with TemporaryDirectory() as temporary_directory:
        video_path = Path(temporary_directory) / "input.mp4"
        video_path.write_bytes(video)
        capture = cv2.VideoCapture(video_path.as_posix())
        try:
            if not capture.isOpened():
                raise ValueError("video must contain a readable echocardiogram")

            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            if frame_total > 0:
                frame_indices = np.linspace(
                    0, frame_total - 1, num=_FRAME_COUNT, dtype=int
                )
                for frame_index in frame_indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    read_successfully, frame = capture.read()
                    if not read_successfully:
                        raise ValueError("video must contain a readable echocardiogram")
                    frames.append(frame)
            else:
                while True:
                    read_successfully, frame = capture.read()
                    if not read_successfully:
                        break
                    frames.append(frame)
                if not frames:
                    raise ValueError("video must contain a readable echocardiogram")
                frame_indices = np.linspace(
                    0, len(frames) - 1, num=_FRAME_COUNT, dtype=int
                )
                frames = [frames[int(index)] for index in frame_indices]
        finally:
            capture.release()

    resized = [
        cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            (_IMAGE_SIZE, _IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        for frame in frames
    ]
    tensor = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).float() / 255
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    standard_deviation = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    tensor = (tensor - mean) / standard_deviation
    return tensor.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)


def _decode_panecho_image(image: bytes, torch: Any) -> Any:
    """Decode and normalize one image for the PanEcho image encoder."""
    from io import BytesIO

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(image)) as source:
            resized = source.convert("RGB").resize(
                (_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(resized, dtype=np.float32) / 255
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("image must contain a readable echocardiogram frame") from exc

    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    standard_deviation = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return ((tensor - mean) / standard_deviation).to(DEVICE)


@lru_cache(maxsize=1)
def _load_panecho_model(model_kind: str) -> Any:
    """Load one PanEcho checkpoint and retain it for subsequent requests."""
    import torch

    model_options: dict[str, dict[str, Any]] = {
        "tasks": {"tasks": "all"},
        "video_embedding": {"backbone_only": True},
        "image_embedding": {"image_encoder_only": True},
    }
    if model_kind not in model_options:
        raise ValueError("model_kind must be a supported PanEcho model variant")

    model = torch.hub.load(
        _PANECHO_REPOSITORY,
        "PanEcho",
        trust_repo=True,
        **model_options[model_kind],
    )
    model = model.to(DEVICE)
    model.eval()
    return model


def _serialize_panecho_predictions(model: Any, outputs: Any) -> dict[str, Any]:
    """Convert PanEcho outputs to JSON-safe, labeled prediction values."""
    task_by_name = {task.task_name: task for task in model.tasks}
    predictions: dict[str, Any] = {}
    for task_name, output in outputs.items():
        task = task_by_name.get(task_name)
        if task is None:
            raise RuntimeError("PanEcho output has no matching task description")

        values = output[0].detach().float().cpu().reshape(-1).tolist()
        if task.task_type == "regression":
            if len(values) != 1:
                raise RuntimeError("PanEcho regression output must be a single value")
            predictions[task_name] = float(values[0])
            continue

        class_names = [str(label) for label in task.class_names]
        if len(class_names) != len(values):
            raise RuntimeError("PanEcho output count does not match its task labels")
        predictions[task_name] = {
            label: float(value)
            for label, value in zip(class_names, values, strict=True)
        }
    return predictions


def _validate_video_predictions(
    video_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the serialized predictions returned by PanEcho inference."""
    if not video_predictions:
        raise ValueError("at least one PanEcho video prediction is required")

    task_names: set[str] | None = None
    for result in video_predictions:
        if not isinstance(result, dict) or result.get("model") != "PanEcho":
            raise ValueError("each result must be a PanEcho video prediction")
        predictions = result.get("predictions")
        if not isinstance(predictions, dict) or not predictions:
            raise ValueError("each result must contain task predictions")

        current_task_names = set(predictions)
        if task_names is None:
            task_names = current_task_names
        elif current_task_names != task_names:
            raise ValueError("all PanEcho results must contain the same tasks")

        for prediction in predictions.values():
            values = (
                prediction.values() if isinstance(prediction, dict) else (prediction,)
            )
            if isinstance(prediction, dict) and not prediction:
                raise ValueError("classification predictions must contain class values")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(value)
                for value in values
            ):
                raise ValueError("PanEcho predictions must contain finite numbers")

    first_predictions = video_predictions[0]["predictions"]
    for result in video_predictions[1:]:
        for task_name, first_value in first_predictions.items():
            current_value = result["predictions"][task_name]
            if isinstance(first_value, dict) != isinstance(current_value, dict):
                raise ValueError("PanEcho task outputs must have consistent types")
            if (
                isinstance(first_value, dict)
                and current_value.keys() != first_value.keys()
            ):
                raise ValueError("PanEcho class labels must match across videos")

    return [result["predictions"] for result in video_predictions]


@ULTRASOUND.register("panecho_predict_video")
@requires(_PANECHO_DEPS, extra="ultrasound")
def panecho_predict_video(video: bytes) -> dict[str, Any]:
    """Predict echocardiography measurements and findings from one video.

    PanEcho returns regression estimates and class probabilities for its
    echocardiography reporting tasks. Use ``panecho_aggregate_study`` to
    combine predictions from multiple videos in the same study.

    Args:
        video: Echocardiography video bytes.

    Returns:
        Model name and a mapping of task names to measurements or class
        probability mappings. Measurements use the units described by PanEcho.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
        https://doi.org/10.1001/jama.2025.8731
    """
    import cv2
    import torch

    tensor = _decode_panecho_video(video, cv2, torch)
    model = _load_panecho_model("tasks")
    with torch.inference_mode():
        outputs = model(tensor)

    return {
        "model": "PanEcho",
        "predictions": _serialize_panecho_predictions(model, outputs),
    }


@ULTRASOUND.register("panecho_aggregate_study")
def panecho_aggregate_study(
    video_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate PanEcho predictions from videos in one echocardiography study.

    Each video prediction receives equal weight when producing the study-level
    result, matching the aggregation described for PanEcho.

    Args:
        video_predictions: Results returned by ``panecho_predict_video`` for
            videos acquired during the same study.

    Returns:
        Model name, number of videos, aggregation method, and task predictions.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
        https://doi.org/10.1001/jama.2025.8731
    """
    predictions = _validate_video_predictions(video_predictions)
    task_names = predictions[0].keys()
    aggregated: dict[str, Any] = {}
    for task_name in task_names:
        first_value = predictions[0][task_name]
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
        "video_count": len(predictions),
        "aggregation": "equal_weight_mean",
        "predictions": aggregated,
    }


@ULTRASOUND.register("panecho_video_embed")
@TASK_EMBED.register("panecho_video_embed")
@requires(_PANECHO_DEPS, extra="ultrasound")
def panecho_video_embed(video: bytes) -> dict[str, Any]:
    """Extract a 768-dimensional PanEcho feature vector from an echo video.

    Args:
        video: Echocardiography video bytes.

    Returns:
        Model name and a JSON-safe 768-dimensional feature vector.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
    """
    import cv2
    import torch

    tensor = _decode_panecho_video(video, cv2, torch)
    model = _load_panecho_model("video_embedding")
    with torch.inference_mode():
        embedding = model(tensor)[0].detach().float().cpu().tolist()

    return {"model": "PanEcho", "embedding": [float(value) for value in embedding]}


@ULTRASOUND.register("panecho_image_embed")
@TASK_EMBED.register("panecho_image_embed")
@requires(_PANECHO_DEPS, extra="ultrasound")
def panecho_image_embed(image: bytes) -> dict[str, Any]:
    """Extract a 768-dimensional PanEcho feature vector from one echo frame.

    Args:
        image: Echocardiography image bytes.

    Returns:
        Model name and a JSON-safe feature vector for the image.

    See Also:
        https://github.com/CarDS-Yale/PanEcho
    """
    import torch

    tensor = _decode_panecho_image(image, torch)
    model = _load_panecho_model("image_embedding")
    with torch.inference_mode():
        embedding = model(tensor)[0].detach().float().cpu().tolist()

    return {"model": "PanEcho", "embedding": [float(value) for value in embedding]}
