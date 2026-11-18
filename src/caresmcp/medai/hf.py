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
from caresmcp.registry import MEDAI, TASK_CLS, TASK_EMB, TASK_RET, TASK_SEG

from ._medsam import MEDSAM_MODEL_ID, medsam_mask

if TYPE_CHECKING:
    from transformers import Sam2Processor

PromptableSegmentationModel = Literal[
    "flaviagiammarino/medsam-vit-base",
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
]

ImageTextRetrievalModel = Literal[
    "google/medsiglip-448",
    "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224",
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


@MEDAI.register("hf_image_text_retrieval")
@TASK_RET.register("hf_image_text_retrieval")
@requires(["torch", "transformers"], extra="hf")
def hf_image_text_retrieval(
    image: bytes,
    texts: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Candidate descriptions in comparison order.",
        ),
    ],
    model_id: Annotated[
        ImageTextRetrievalModel, Field(description=HFMODEL_ID_DESC)
    ] = "google/medsiglip-448",
) -> dict[str, Any]:
    """Rank candidate texts against one image with a dual-encoder model.

    Use a medical model such as MedSigLIP, or a public CLIP or SigLIP
    baseline, to rank free-form descriptions for retrieval. When the
    candidates are class names, the scores also provide zero-shot label
    ranking. MedSigLIP access requires accepting the Health AI Developer
    Foundations terms. Scores are uncalibrated cosine similarities, not
    probabilities or diagnostic evidence.

    Returns:
        Model provenance and text/cosine-similarity pairs in input order.

    See Also:
        https://huggingface.co/google/medsiglip-448
        https://huggingface.co/docs/transformers/model_doc/clip
        https://huggingface.co/docs/transformers/model_doc/siglip
    """
    import torch
    from torch.nn import functional as F

    if not texts or any(not text.strip() for text in texts):
        raise ValueError("texts must contain at least one non-empty description")

    model, processor = get_hf_model_and_processor(model_id, HFModelType.AUTO)
    model = cast(Any, model)
    image_inputs = processor(
        images=ImageOps.exif_transpose(decode_as(image, mode="RGB")),
        return_tensors="pt",
    )
    image_inputs = {name: value.to(DEVICE) for name, value in image_inputs.items()}
    # SigLIP-style encoders pool a fixed final position, so pad every text to
    # the tokenizer's maximum length as their published preprocessing does.
    text_inputs = processor(
        text=texts, padding="max_length", truncation=True, return_tensors="pt"
    )
    text_inputs = {name: value.to(DEVICE) for name, value in text_inputs.items()}
    try:
        with torch.inference_mode():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "model_id must provide Transformers image and text feature methods"
        ) from exc

    if (
        image_features.ndim != 2
        or image_features.shape[0] != 1
        or text_features.ndim != 2
        or text_features.shape[0] != len(texts)
        or not torch.isfinite(image_features).all()
        or not torch.isfinite(text_features).all()
        or (image_features.norm(dim=-1) == 0).any()
        or (text_features.norm(dim=-1) == 0).any()
    ):
        raise RuntimeError("dual encoder returned invalid features")

    image_vector = F.normalize(image_features.float(), dim=-1)[0]
    text_vectors = F.normalize(text_features.float(), dim=-1)
    scores = (text_vectors @ image_vector).clamp(-1.0, 1.0).detach().cpu().tolist()
    return {
        "model_id": model_id,
        "score_type": "cosine_similarity",
        "matches": [
            {"text": text, "score": float(score)}
            for text, score in zip(texts, scores, strict=True)
        ],
    }


def _promptable_pipeline(
    model_id: PromptableSegmentationModel,
) -> tuple[HFModelType, bool]:
    """Return the loader type and mask-selection mode for one checkpoint.

    MedSAM checkpoints use the official square-canvas, per-image min-max
    preprocessing and a single-mask output. SAM2.1 checkpoints use the fixed
    SAM2 processor and multi-mask output with predicted-IoU selection.
    """
    if model_id == MEDSAM_MODEL_ID:
        return HFModelType.SAM, False
    if model_id.startswith("facebook/sam2.1-"):
        return HFModelType.SAM2, True
    raise ValueError("unsupported promptable segmentation model_id")


def promptable_segmentation_mask(
    source: Image.Image,
    box: tuple[float, float, float, float],
    model_id: PromptableSegmentationModel,
) -> np.ndarray:
    """Return an original-size binary mask for one RGB image and box prompt.

    MedSAM follows its official 1024x1024 square resize and per-image min-max
    normalization. SAM2.1 uses its Hub processor and predicted-IoU mask
    selection.

    Args:
        source: RGB image containing the prompted region.
        box: Prompt box in original pixels as ``(x1, y1, x2, y2)``.
        model_id: Promptable checkpoint selecting the model pipeline.

    Returns:
        A boolean foreground mask at the source image size.
    """
    import torch

    model_type, multimask_output = _promptable_pipeline(model_id)
    if model_type is HFModelType.SAM:
        return medsam_mask(source, box, model_id)

    width, height = source.size
    x1, y1, x2, y2 = check_bbox(box, width, height)
    model, processor = get_hf_model_and_processor(model_id, model_type)
    inputs = processor(
        images=source, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt"
    )
    inputs = {
        name: value.to(DEVICE) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=multimask_output)
    masks = cast("Sam2Processor", processor).post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )
    best = int(outputs.iou_scores.detach().float().cpu()[0, 0].argmax())
    mask = np.asarray(masks[0][0, best].numpy() > 0.0)
    if mask.shape != (height, width) or not mask.any():
        raise RuntimeError("Segmenter returned an empty or mis-sized mask")
    return mask


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
    ] = "flaviagiammarino/medsam-vit-base",
) -> bytes:
    """Segment a box-prompted region in one medical image (white foreground PNG).

    MedSAM uses the official square-canvas, per-image min-max preprocessing of
    its July 2023 release; SAM2.1 checkpoints are general-purpose. The mask
    follows the prompted box and requires review.
    """
    source = ImageOps.exif_transpose(decode_as(image, mode="RGB"))
    mask = promptable_segmentation_mask(source, box, model_id)
    return encode_as(Image.fromarray(mask.astype(np.uint8) * 255))
