"""MONAI tumor detection and nuclei analysis for pathology image regions."""

from io import BytesIO
from typing import Annotated, Any, cast

import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as, get_classification_scores
from caresmcp.api.models import get_monai_model
from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TAG_MONAI, TASK_CLS, TASK_SEG

TUMOR_PATCH_SIZE = 224
TUMOR_MODEL_ID = "MONAI/pathology_tumor_detection"
NUCLEI_CLASSIFIER_ID = "MONAI/pathology_nuclei_classification"
HOVERNET_ID = "MONAI/pathology_nuclei_segmentation_classification"
NUCLEI_CLASSES = ("other", "inflammatory", "epithelial", "spindle_shaped")


@PATHOLOGY.register("monai_pathology_tumor_detection")
@TAG_MONAI.register("monai_pathology_tumor_detection")
@TASK_CLS.register("monai_pathology_tumor_detection")
@requires(["torch", "torchvision", "monai", "huggingface_hub"], extra="monai")
def monai_pathology_tumor_detection(
    image: Annotated[
        bytes,
        Field(
            description="Encoded RGB H&E lymph-node tile or ROI at level-zero resolution."
        ),
    ],
    batch_size: Annotated[
        int, Field(description="Number of tumor patches scored together.", ge=1, le=400)
    ] = 32,
) -> dict[str, Any]:
    """Score tumor-like content in an H&E lymph-node tile or ROI.

    Returns:
        A score from 0 to 1 for each 224×224 patch, its [left, top, right,
        bottom] pixel box, and the highest patch score. Scores are not calibrated
        probabilities.

    See Also:
        https://huggingface.co/MONAI/pathology_tumor_detection
        https://github.com/Project-MONAI/model-zoo/blob/dev/models/pathology_tumor_detection/docs/README.md
        https://arxiv.org/abs/1512.03385
    """
    import torch

    model_id = TUMOR_MODEL_ID
    pixels = np.asarray(decode_as(image, mode="RGB"))
    height, width = pixels.shape[:2]
    boxes = [
        [x, y, min(x + TUMOR_PATCH_SIZE, width), min(y + TUMOR_PATCH_SIZE, height)]
        for y in range(0, height, TUMOR_PATCH_SIZE)
        for x in range(0, width, TUMOR_PATCH_SIZE)
    ]
    model = get_monai_model(model_id)
    patches = []
    with torch.inference_mode():
        for offset in range(0, len(boxes), batch_size):
            batch_boxes = boxes[offset : offset + batch_size]
            batch = np.full(
                (len(batch_boxes), TUMOR_PATCH_SIZE, TUMOR_PATCH_SIZE, 3),
                255,
                dtype=np.float32,
            )
            for index, (left, top, right, bottom) in enumerate(batch_boxes):
                batch[index, : bottom - top, : right - left] = pixels[
                    top:bottom, left:right
                ]
            inputs = torch.from_numpy(batch).permute(0, 3, 1, 2).to(DEVICE)
            logits = model(inputs / 127.5 - 1.0)
            if logits.numel() != len(batch_boxes) or not torch.isfinite(logits).all():
                raise RuntimeError("MONAI returned invalid patch tumor logits")
            scores = logits.sigmoid().reshape(-1).cpu().tolist()
            patches.extend(
                {"box": box, "tumor_score": score}
                for box, score in zip(batch_boxes, scores, strict=True)
            )
    return {
        "model_id": model_id,
        "image_size": {"width": width, "height": height},
        "patch_size": TUMOR_PATCH_SIZE,
        "max_tumor_score": max(patch["tumor_score"] for patch in patches),
        "patches": patches,
    }


@PATHOLOGY.register("monai_pathology_nuclei_classification")
@TAG_MONAI.register("monai_pathology_nuclei_classification")
@TASK_CLS.register("monai_pathology_nuclei_classification")
@requires(["torch", "monai", "huggingface_hub"], extra="monai")
def monai_pathology_nuclei_classification(
    image: Annotated[
        bytes,
        Field(
            description="Encoded 128x128 RGB H&E patch centered on a nucleus at CoNSeP's 40x scale."
        ),
    ],
    mask: Annotated[
        bytes,
        Field(
            description="Encoded 128x128 grayscale or indexed mask: zero background, nonzero target nucleus."
        ),
    ],
) -> dict[str, Any]:
    """Classify the nucleus marked in a colorectal tissue patch.

    Returns:
        The predicted class and scores for other, inflammatory, epithelial,
        and spindle-shaped nuclei. Scores sum to 1 but are not calibrated
        probabilities.

    See Also:
        https://huggingface.co/MONAI/pathology_nuclei_classification
        https://github.com/Project-MONAI/model-zoo/blob/dev/models/pathology_nuclei_classification/docs/README.md
    """
    import torch
    from monai.apps.nuclick.transforms import AddLabelAsGuidanced

    model_id = NUCLEI_CLASSIFIER_ID
    pixels = np.asarray(decode_as(image, mode="RGB"), dtype=np.float32)
    with Image.open(BytesIO(mask)) as decoded_mask:
        if decoded_mask.mode not in ("1", "L", "I", "I;16", "P"):
            raise ValueError("mask must be a grayscale or indexed nucleus mask")
        labels = np.asarray(decoded_mask).copy()
    if pixels.shape != (128, 128, 3) or labels.shape != (128, 128):
        raise ValueError("image and mask must both be 128x128 pixels")
    if not np.any(labels > 0):
        raise ValueError("mask must contain a target nucleus")
    data = AddLabelAsGuidanced(keys="image", source="label")(
        # Match MONAI PILReader's default spatial indexing (channels, x, y).
        {"image": pixels.transpose(2, 1, 0) / 127.5 - 1.0, "label": labels.T.copy()}
    )
    inputs = torch.as_tensor(data["image"]).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = get_monai_model(model_id)(inputs)
        if logits.shape != (1, 4) or not torch.isfinite(logits).all():
            raise RuntimeError("MONAI returned invalid nucleus class logits")
        class_scores = get_classification_scores(
            logits, dict(enumerate(NUCLEI_CLASSES)), "softmax"
        )
    return {
        "model_id": model_id,
        "predicted_class": max(class_scores, key=class_scores.__getitem__),
        "class_scores": class_scores,
    }


@PATHOLOGY.register("monai_pathology_nuclei_segmentation_classification")
@TAG_MONAI.register("monai_pathology_nuclei_segmentation_classification")
@TASK_SEG.register("monai_pathology_nuclei_segmentation_classification")
@TASK_CLS.register("monai_pathology_nuclei_segmentation_classification")
@requires(["torch", "monai", "huggingface_hub", "skimage", "scipy"], extra="monai")
def monai_pathology_nuclei_segmentation_classification(
    image: Annotated[
        bytes,
        Field(description="Encoded RGB H&E tissue tile or ROI at CoNSeP's 40x scale."),
    ],
    batch_size: Annotated[
        int,
        Field(description="Number of HoVerNet windows scored together.", ge=1, le=16),
    ] = 4,
) -> dict[str, Any]:
    """Segment nuclei and assign a type to each in colorectal tissue.

    Returns:
        The nucleus count and maps aligned with the input as [y][x]. Positive
        IDs in ``instance_map`` identify nuclei. ``type_map`` uses 0 for
        background, then 1–4 for other, inflammatory, epithelial, and
        spindle-shaped nuclei.

    See Also:
        https://huggingface.co/MONAI/pathology_nuclei_segmentation_classification
        https://github.com/Project-MONAI/model-zoo/blob/dev/models/pathology_nuclei_segmentation_classification/docs/README.md
        https://doi.org/10.1016/j.media.2019.101563
    """
    import torch
    from monai.apps.pathology.inferers import SlidingWindowHoVerNetInferer
    from monai.apps.pathology.transforms import (
        HoVerNetInstanceMapPostProcessing,
        HoVerNetNuclearTypePostProcessing,
    )

    model_id = HOVERNET_ID
    pixels = np.asarray(decode_as(image, mode="RGB"), dtype=np.float32)
    # The released pipeline uses PILReader with reverse_indexing=True.
    inputs = torch.from_numpy(pixels.transpose(2, 1, 0).copy()).unsqueeze(0) / 255.0
    inferer = SlidingWindowHoVerNetInferer(
        roi_size=256,
        sw_batch_size=batch_size,
        overlap=1.0 - 164.0 / 256.0,
        padding_mode="constant",
        cval=0,
        # MONAI annotates this as a one-element tuple, but requires four pads.
        extra_input_padding=(46, 46, 46, 46),  # ty: ignore[invalid-argument-type]
        sw_device=DEVICE,
        device="cpu",
    )
    with torch.inference_mode():
        predictions = cast(
            dict[str, torch.Tensor], inferer(inputs, get_monai_model(model_id))
        )
    info, instance_map = HoVerNetInstanceMapPostProcessing(
        sobel_kernel_size=21, marker_threshold=0.4, marker_radius=2
    )(predictions["nucleus_prediction"][0], predictions["horizontal_vertical"][0])
    _, type_map = HoVerNetNuclearTypePostProcessing()(
        predictions["type_prediction"][0], info, instance_map
    )
    if type_map is None:
        raise RuntimeError("MONAI did not return a nucleus type map")
    instances = (
        np.asarray(torch.as_tensor(instance_map).cpu()).squeeze(0).T.astype(np.int32)
    )
    types = np.asarray(torch.as_tensor(type_map).cpu()).squeeze(0).T.astype(np.uint8)
    if instances.shape != pixels.shape[:2] or types.shape != pixels.shape[:2]:
        raise RuntimeError("MONAI returned nucleus maps with unexpected dimensions")
    return {
        "model_id": model_id,
        "labels": dict(enumerate(("background", *NUCLEI_CLASSES))),
        "instance_count": int(np.unique(instances[instances > 0]).size),
        "instance_map": instances.tolist(),
        "type_map": types.tolist(),
    }
