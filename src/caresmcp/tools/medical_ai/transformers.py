from typing import Any, Literal

from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI
from caresmcp.util.moduleloader import requires

from ..image_processing import decode_rgb


@MEDAI.register("classify_with_transformers")
@requires(["torch", "transformers"], extra="hf")
def classify_with_transformers(
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
        revision: Optional model revision or commit identifier.
        local_files_only: Whether to use only locally cached checkpoint files.
        score_mode: Apply sigmoid for independent labels or softmax for
            mutually exclusive classes.

    Returns:
        Model provenance, score mode, and scores keyed by checkpoint labels.

    See Also:
        https://huggingface.co/docs/transformers/tasks/image_classification
    """
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        ProcessorMixin,
    )

    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if score_mode not in ("sigmoid", "softmax"):
        raise ValueError("score_mode must be 'sigmoid' or 'softmax'")

    processor: ProcessorMixin = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id).to(DEVICE)
    model.eval()

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
        "model_id": model_id,
        "score_mode": score_mode,
        "scores": {
            label: float(score)
            for label, score in zip(labels, score_values, strict=True)
        },
    }
