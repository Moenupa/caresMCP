from enum import Enum
from functools import lru_cache
from typing import Any, Literal

from caresmcp.constants import DEVICE, LRU_MAX_SIZE
from caresmcp.registry import MEDAI
from caresmcp.tools.image_processing import decode_rgb
from caresmcp.util.moduleloader import requires


class HFModelType(Enum):
    """Transformers model and processor class combinations used internally."""

    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_ENCODER = "image_encoder"
    AUTO = "auto"


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
    """
    match model_type:
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
