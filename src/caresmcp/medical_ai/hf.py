from collections import Counter
from typing import Any, Literal

from caresmcp.api.imageio import decode_rgb
from caresmcp.api.models import get_hf_model_and_processor
from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TASK_CLS
from caresmcp.util.moduleloader import requires


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
        Scores keyed by checkpoint labels. Repeated label names include their
        class IDs to distinguish separate classes.

    See Also:
        https://huggingface.co/docs/transformers/tasks/image_classification
    """
    import torch

    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if score_mode not in ("sigmoid", "softmax"):
        raise ValueError("score_mode must be 'sigmoid' or 'softmax'")

    model, processor = get_hf_model_and_processor(model_id)
    config_labels = model.config.id2label
    if config_labels is None:
        raise ValueError(
            "Checkpoint has no id2label config, consider using a different model or checkpoint."
        )

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

    labels = [
        str(config_labels.get(index, config_labels.get(str(index), f"class_{index}")))
        for index in range(len(score_values))
    ]
    label_counts = Counter(labels)
    labels = [
        f"{label} (class {index})" if label_counts[label] > 1 else label
        for index, label in enumerate(labels)
    ]
    return {
        label: float(score) for label, score in zip(labels, score_values, strict=True)
    }
