from collections import Counter
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
from PIL import Image, ImageOps
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import check_bbox, decode_as, encode_as, get_classification_scores
from caresmcp.api.models import (
    HFMODEL_ID_DESC,
    SOFTMAX_DESC,
    HFModelType,
    get_hf_model_and_processor,
)
from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TASK_CLS, TASK_EMB, TASK_SEG

if TYPE_CHECKING:
    from transformers import Sam2Processor, SamProcessor

PromptableSegmentationModel = Literal[
    "wanglab/medsam-vit-base",
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
]


@MEDAI.register("hf_classifier")
@TASK_CLS.register("hf_classifier")
@requires(["torch", "transformers"], extra="hf")
def hf_classifier(
    image: bytes,
    model_id: Annotated[str, Field(description=HFMODEL_ID_DESC)],
    score_mode: Annotated[
        Literal["sigmoid", "softmax"], Field(description=SOFTMAX_DESC)
    ] = "sigmoid",
) -> dict[str, Any]:
    """Run a trained Transformers image-classification head.

    See Also:
        https://huggingface.co/docs/transformers/tasks/image_classification
    """
    import torch

    model, processor = get_hf_model_and_processor(
        model_id, HFModelType.IMAGE_CLASSIFICATION
    )
    config_labels = model.config.id2label
    if config_labels is None:
        raise ValueError(
            "Checkpoint has no id2label config, consider using a different model or checkpoint."
        )

    inputs = processor(images=decode_as(image, mode="RGB"), return_tensors="pt")
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits

    labels = [
        str(config_labels.get(index, config_labels.get(str(index), f"class_{index}")))
        for index in range(logits.shape[-1])
    ]
    label_counts = Counter(labels)
    labels = [
        f"{label} (class {index})" if label_counts[label] > 1 else label
        for index, label in enumerate(labels)
    ]
    return get_classification_scores(logits, dict(enumerate(labels)), score_mode)


@MEDAI.register("hf_image_embed")
@TASK_EMB.register("hf_image_embed")
@requires(["torch", "transformers"], extra="hf")
def hf_image_embed(
    image: bytes,
    model_id: Annotated[
        str, Field(description=HFMODEL_ID_DESC)
    ] = "facebook/dinov2-base",
    pooling: Annotated[
        Literal["cls", "mean"],
        Field(description="First token or mean of non-first tokens."),
    ] = "cls",
) -> dict[str, Any]:
    """Extract image features with a Hugging Face image encoder.

    For token outputs, mean pooling excludes the first token;
    for pooled outputs, pooling is supplied by the model.
    """
    import torch

    model, processor = get_hf_model_and_processor(model_id, HFModelType.IMAGE_ENCODER)
    inputs = processor(images=decode_as(image, mode="RGB"), return_tensors="pt")
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)

    tokens = getattr(output, "last_hidden_state", None)
    if tokens is not None and tokens.ndim == 3 and tokens.shape[0] == 1:
        if tokens.shape[1] == 0 or tokens.shape[2] == 0:
            raise ValueError("encoder returned an empty last_hidden_state")
        if pooling == "cls" or tokens.shape[1] == 1:
            vector = tokens[:, 0, :]
            pooling_used = "cls"
        else:
            vector = tokens[:, 1:, :].mean(dim=1)
            pooling_used = "mean"
    else:
        vector = getattr(output, "pooler_output", None)
        if vector is None or vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError(
                "encoder must return a single-item last_hidden_state or pooler_output"
            )
        pooling_used = "pooler"
    embedding = vector[0].detach().float().cpu().tolist()
    return {"model_id": model_id, "pooling": pooling_used, "embedding": embedding}


@MEDAI.register("hf_segment_image")
@TASK_SEG.register("hf_segment_image")
@requires(["torch", "torchvision", "transformers"], extra="hf")
def hf_segment_image(
    image: bytes,
    box: Annotated[
        tuple[float, float, float, float],
        Field(description="Prompt box in pixels as [x1, y1, x2, y2]."),
    ],
    model_id: Annotated[
        PromptableSegmentationModel, Field(description=HFMODEL_ID_DESC)
    ] = "wanglab/medsam-vit-base",
) -> bytes:
    """Segment a box-prompted region in one medical image (white foreground PNG).

    MedSAM is a general medical segmenter; SAM2.1 checkpoints are general-purpose.
    The mask follows the prompted box and requires review.
    """
    import torch

    is_sam2 = model_id.startswith("facebook/sam2.1-")
    model, processor = get_hf_model_and_processor(
        model_id, HFModelType.SAM2 if is_sam2 else HFModelType.SAM
    )
    source = ImageOps.exif_transpose(decode_as(image, mode="RGB"))
    width, height = source.size
    x1, y1, x2, y2 = check_bbox(box, width, height)
    inputs = processor(
        images=source,
        input_boxes=[[[x1, y1, x2, y2]]],
        return_tensors="pt",
    )
    inputs = {
        name: value.to(DEVICE) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**inputs)
    if is_sam2:
        masks = cast("Sam2Processor", processor).post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )
    else:
        masks = cast("SamProcessor", processor).post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
    best = int(outputs.iou_scores.detach().float().cpu()[0, 0].argmax())
    mask = np.asarray(masks[0][0, best].numpy() > 0.0)
    if mask.shape != (height, width) or not mask.any():
        raise RuntimeError("Segmenter returned an empty or mis-sized mask")
    return encode_as(Image.fromarray(mask.astype(np.uint8) * 255))
