"""Detection on pediatric wrist radiographs with a portable YOLOv7 release."""

from typing import Annotated

import numpy as np
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.models.wrist import get_wrist_onnx_model
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
    original-image pixel coordinates. The upstream model is released under GPL-3.0.

    See Also:
        https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection
    """
    import cv2

    bgr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode wrist radiograph image")
    pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = pixels.shape[:2]
    # Follow the released ONNX script's direct resize, RGB and [0, 1] scaling.
    resized = cv2.resize(pixels, (640, 640), interpolation=cv2.INTER_LINEAR)
    tensor = (resized.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    session = get_wrist_onnx_model()
    rows = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detections = []
    for row in rows:
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
