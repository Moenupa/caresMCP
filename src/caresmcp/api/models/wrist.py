"""Cached portable pediatric wrist detector."""

from functools import lru_cache
from typing import Any

from .download import get_checkpoint_path

WRIST_MODEL_URL = (
    "https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection/releases/"
    "download/trained-models/yolov7-p6-bonefracture.onnx"
)
WRIST_MODEL_SHA256 = "bf65ea16ed4a6b7cc911e60420100aeae0eb748eb8ab8ff7e88544c7c00a9320"


@lru_cache(maxsize=1)
def get_wrist_onnx_model() -> Any:
    """Download the fixed ONNX release and cache its CPU inference session."""
    import onnxruntime

    checkpoint = get_checkpoint_path(
        WRIST_MODEL_URL, "yolov7-p6-bonefracture.onnx", WRIST_MODEL_SHA256
    )
    return onnxruntime.InferenceSession(
        str(checkpoint), providers=["CPUExecutionProvider"]
    )
