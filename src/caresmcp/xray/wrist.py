"""Detection on pediatric wrist radiographs with a portable YOLOv7 release."""

from typing import Annotated

import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as
from caresmcp.api.models import get_onnx_session
from caresmcp.registry import TASK_LOC, XRAY

WRIST_LABELS = (
    "boneanomaly",
    "bonelesion",
    "foreignbody",
    "fracture",
    "metal",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
    "text",
)
WRIST_MODEL_URL = (
    "https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection/releases/"
    "download/trained-models/yolov7-p6-bonefracture.onnx"
)
WRIST_MODEL_FILENAME = "yolov7-p6-bonefracture.onnx"
WRIST_MODEL_SHA256 = "bf65ea16ed4a6b7cc911e60420100aeae0eb748eb8ab8ff7e88544c7c00a9320"


def _decode_wrist_rgb(image: bytes) -> np.ndarray:
    """Return 8-bit RGB pixels, scaling 16-bit radiographs like OpenCV.

    GRAZPEDWRI-DX releases 16-bit PNGs. Converting ``I;16`` directly to RGB
    clips values above 255 instead of rescaling, which erases most of the
    radiograph's contrast; the released detector code read these files with
    OpenCV, whose 8-bit conversion keeps the high byte (``value >> 8``).
    """
    source = decode_as(image)
    if source.mode.startswith("I;16"):
        pixels = np.asarray(source).astype(np.uint16)
        source = Image.fromarray(pixels >> 8)
    return np.asarray(source.convert("RGB"), dtype=np.uint8)


@XRAY.register("detect_pediatric_wrist_findings")
@TASK_LOC.register("detect_pediatric_wrist_findings")
@requires(["torch", "onnxruntime"], extra="xray")
def detect_pediatric_wrist_findings(
    image: Annotated[bytes, Field(description="Pediatric wrist radiograph image")],
    score_threshold: Annotated[
        float, Field(ge=0, le=1, description="Minimum detection confidence to return")
    ] = 0.3,
) -> dict:
    """Detect fractures and associated findings on pediatric wrist radiographs.

    Use this to identify and locate candidate findings in a pediatric wrist
    image. Adjust the score threshold to control which detections are returned.
    The detector supports nine GRAZPEDWRI-DX classes; fracture labels do not
    specify bone, morphology, or Salter–Harris type. Bounding boxes use
    original-image pixel coordinates.

    See Also:
        https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection
    """
    import cv2

    pixels = _decode_wrist_rgb(image)
    height, width = pixels.shape[:2]
    # Follow the released ONNX script's direct resize, RGB and [0, 1] scaling.
    resized = cv2.resize(pixels, (640, 640), interpolation=cv2.INTER_LINEAR)
    tensor = (resized.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    session = get_onnx_session(
        WRIST_MODEL_URL, WRIST_MODEL_FILENAME, WRIST_MODEL_SHA256
    )
    rows = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detections = []
    for row in rows:  # ty: ignore[not-iterable]
        left, top, right, bottom, score, label = row[:6]
        if score < score_threshold:
            continue
        box = np.asarray([left, top, right, bottom]) * np.asarray(
            [width / 640, height / 640, width / 640, height / 640]
        )
        box = np.clip(box, 0, [width, height, width, height])
        detections.append(
            {
                "label": WRIST_LABELS[int(label)],
                "score": round(float(score), 4),
                "box_xyxy": box.round(2).tolist(),
            }
        )
    return {
        "model_id": "mdciri/YOLOv7-Bone-Fracture-Detection",
        "checkpoint": "trained-models/yolov7-p6-bonefracture.onnx",
        "image_width": width,
        "image_height": height,
        "detections": detections,
    }
