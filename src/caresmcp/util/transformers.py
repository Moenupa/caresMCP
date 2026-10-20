from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import Any, Literal

from caresmcp.constants import DEVICE, LRU_MAX_SIZE
from caresmcp.registry import MEDAI, TASK_CLS
from caresmcp.tools.image_processing import decode_rgb
from caresmcp.util.moduleloader import requires


class HFModelType(Enum):
    """Transformers model and processor class combinations used internally."""

    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_ENCODER = "image_encoder"
    AUTO = "auto"
    SAM = "sam"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_hf_model_and_processor(
    model_id: str,
    model_type: HFModelType = HFModelType.IMAGE_CLASSIFICATION,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
) -> tuple[Any, Any]:
    """Load and cache a Transformers model and its processor.

    Image-classification checkpoints use their task-specific model and image
    processor classes by default. Image encoders use ``AutoModel`` with
    ``AutoImageProcessor``. ``HFModelType.AUTO`` uses ``AutoModel`` and
    ``AutoProcessor`` for checkpoints with custom model interfaces.

    Never support revision or local_files_only. Since it brings complexity we/agents dont need.
    """
    match model_type:
        case HFModelType.SAM:
            from transformers import SamModel, SamProcessor

            model_class = SamModel
            processor_class = SamProcessor
        case HFModelType.SEMANTIC_SEGMENTATION:
            from transformers import (
                AutoImageProcessor,
                AutoModelForSemanticSegmentation,
            )

            model_class = AutoModelForSemanticSegmentation
            processor_class = AutoImageProcessor
        case HFModelType.IMAGE_CLASSIFICATION:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            model_class = AutoModelForImageClassification
            processor_class = AutoImageProcessor
        case HFModelType.IMAGE_ENCODER:
            from transformers import AutoImageProcessor, AutoModel

            model_class = AutoModel
            processor_class = AutoImageProcessor
        case HFModelType.AUTO:
            from transformers import AutoModel, AutoProcessor

            model_class = AutoModel
            processor_class = AutoProcessor
        case _:
            raise ValueError("model_type must be an HFModelType member")

    model_kwargs: dict[str, Any] = {}
    processor_kwargs: dict[str, Any] = {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
        processor_kwargs["trust_remote_code"] = True
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    processor = processor_class.from_pretrained(model_id, **processor_kwargs)
    model = model_class.from_pretrained(model_id, **model_kwargs).to(DEVICE)
    model.eval()
    return model, processor


@MEDAI.register("hf_classifier")
@TASK_CLS.register("hf_classifier")
@requires(["torch", "transformers"], extra="hf")
def hf_classifier(
    image: bytes,
    model_id: str,
    score_mode: Literal["sigmoid", "softmax"] = "sigmoid",
) -> dict[str, Any]:
    """Run a trained Transformers image-classification head on a CXR.

    Use a checkpoint with both a compatible image processor and a trained
    classification head. Encoder-only RAD-DINO or CXFormer checkpoints do not
    provide disease predictions until a task-specific head is trained. Scores
    are model outputs, not calibrated clinical probabilities.

    Args:
        image: Chest radiograph image bytes.
        model_id: Local model directory or Hugging Face model ID.
        score_mode: Apply sigmoid for independent labels or softmax for
            mutually exclusive classes.

    Returns:
        Model provenance, score mode, and scores keyed by checkpoint labels.

    See Also:
        https://huggingface.co/docs/transformers/tasks/image_classification
    """
    import torch

    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if score_mode not in ("sigmoid", "softmax"):
        raise ValueError("score_mode must be 'sigmoid' or 'softmax'")

    model, processor = _get_hf_model_and_processor(model_id)

    inputs = processor(images=decode_rgb(image), return_tensors="pt")
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        scores = (
            torch.sigmoid(logits)
            if score_mode == "sigmoid"
            else torch.softmax(logits, dim=-1)
        )
        score_values = scores.detach().float().cpu().tolist()

    config_labels = model.config.id2label
    labels = [
        str(config_labels.get(index, config_labels.get(str(index), f"class_{index}")))
        for index in range(len(score_values))
    ]
    if len(set(labels)) != len(labels):
        raise ValueError("checkpoint id2label contains duplicate labels")
    return {
        label: float(score) for label, score in zip(labels, score_values, strict=True)
    }


def encode_label_mask(labels: Any) -> bytes:
    """Encode integer class IDs as a palette PNG with transparent background.

    Masks deliberately preserve class IDs rather than multiplying patient
    intensities by labels. This exception supports downstream measurements.
    """
    import numpy as np
    from PIL import Image

    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.isfinite(labels).all():
        raise RuntimeError("segmenter must return a finite 2D label map")
    if (
        (labels < 0).any()
        or (labels > 255).any()
        or (labels != labels.astype(int)).any()
    ):
        raise RuntimeError("segmenter labels must be integers from 0 through 255")
    mask = Image.fromarray(labels.astype(np.uint8)).convert("P")
    # Keep the existing cardiac label colors; extend deterministically.
    colors = [(0, 0, 0), (31, 119, 180), (255, 127, 14), (44, 160, 44)]
    colors.extend(
        ((i * 67) % 256, (i * 131) % 256, (i * 197) % 256) for i in range(4, 256)
    )
    mask.putpalette([value for color in colors for value in color])
    output = BytesIO()
    mask.save(output, format="PNG", transparency=0)
    return output.getvalue()
