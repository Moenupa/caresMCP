"""Prompted ultrasound segmentation with the pretrained SAM 2 model."""

import math
from typing import Annotated, Any, Literal, cast, get_args

import numpy as np
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as, encode_label_mask
from caresmcp.api.models import HFModelType, get_hf_model_and_processor
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_SEG, ULTRASOUND

Sam2ModelId = Literal[
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
]
US_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
_SAM2_MODEL_IDS = get_args(Sam2ModelId)


def _validate_sam2_prompts(
    points: list[list[float]] | None,
    point_labels: list[int] | None,
    box: list[float] | None,
    width: int,
    height: int,
) -> None:
    """Validate prompt geometry on the original image's pixel grid."""
    if point_labels is not None and points is None:
        raise ValueError("point_labels requires points")
    if box is None and not points:
        raise ValueError("provide a box or at least one labeled point")
    if box is not None:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("box must contain four finite XYXY coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("box must have positive area and lie within the image")
    if points is not None:
        if not points or point_labels is None or len(points) != len(point_labels):
            raise ValueError(
                "points and point_labels must have equal, non-zero lengths"
            )
        if any(label not in (0, 1) for label in point_labels):
            raise ValueError("point_labels must contain only 0 or 1")
        for point in points:
            if len(point) != 2 or not all(math.isfinite(value) for value in point):
                raise ValueError("each point must contain two finite coordinates")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("point coordinates must lie within the image")


@ULTRASOUND.register("segment_ultrasound_sam2")
@TASK_SEG.register("segment_ultrasound_sam2")
@requires(["torch", "torchvision", "transformers"], extra="ultrasound")
def segment_ultrasound_sam2(
    image: Annotated[bytes, Field(description="Ultrasound image or frame.")],
    points: Annotated[
        list[list[float]] | None,
        Field(
            description="Optional [x, y] point prompts in original image pixels, applied to one selected structure."
        ),
    ] = None,
    point_labels: Annotated[
        list[int] | None,
        Field(
            description="Matching point labels: 1 for the target structure and 0 for background."
        ),
    ] = None,
    box: Annotated[
        list[float] | None,
        Field(
            description="Optional [x1, y1, x2, y2] box around the target in original image pixels."
        ),
    ] = None,
    model_id: Annotated[
        Sam2ModelId,
        Field(
            description="SAM 2 checkpoint; larger variants trade speed for accuracy."
        ),
    ] = US_SAM2_MODEL_ID,
) -> bytes:
    """Segment one user-selected structure in an ultrasound image with SAM 2.

    Positive and negative points or a box direct the segmentation. The mask
    follows those prompts and does not identify, grade, or measure pathology.
    This is the general-domain SAM 2 checkpoint, not an ultrasound-adapted
    model; its accuracy on a target structure is not established here.

    Returns:
        PNG mask at the input image size: 0 is transparent background and
        1 (blue) is the prompted structure.

    See Also:
        https://github.com/facebookresearch/sam2
        https://huggingface.co/docs/transformers/model_doc/sam2
        https://arxiv.org/abs/2408.00714
    """
    import torch

    if model_id not in _SAM2_MODEL_IDS:
        raise ValueError("model_id must be a supported SAM 2 checkpoint")
    source = decode_as(image, mode="RGB")
    _validate_sam2_prompts(points, point_labels, box, source.width, source.height)

    model, processor = get_hf_model_and_processor(model_id, HFModelType.SAM2)
    processor = cast(Any, processor)
    inputs = processor(
        images=source,
        input_points=None if points is None else [[points]],
        input_labels=None if point_labels is None else [[point_labels]],
        input_boxes=None if box is None else [[box]],
        return_tensors="pt",
    ).to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
        if outputs.pred_masks.ndim != 5 or outputs.pred_masks.shape[:3] != (1, 1, 1):
            raise RuntimeError("SAM 2 must return one mask for the prompted structure")
        if not torch.isfinite(outputs.pred_masks).all():
            raise RuntimeError("SAM 2 returned non-finite mask logits")
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )

    mask = np.asarray(masks[0][0, 0], dtype=np.uint8)
    if mask.shape != (source.height, source.width):
        raise RuntimeError("SAM 2 mask must match the input image size")
    return encode_label_mask(mask)
