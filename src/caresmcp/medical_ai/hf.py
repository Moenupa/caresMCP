from collections import Counter
from typing import Annotated, Any, Literal

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as, get_classification_scores
from caresmcp.api.models import (
    HFMODEL_ID_DESC,
    SOFTMAX_DESC,
    get_hf_model_and_processor,
)
from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TASK_CLS


@MEDAI.register("hf_classifier")
@TASK_CLS.register("hf_classifier")
@requires(["torch", "transformers"], extra="hf")
def hf_classifier(
    image: Annotated[bytes, Field(description="Image to classify")],
    model_id: Annotated[str, Field(description=HFMODEL_ID_DESC)],
    score_mode: Annotated[
        Literal["sigmoid", "softmax"], Field(description=SOFTMAX_DESC)
    ] = "sigmoid",
) -> dict[str, Any]:
    """Run a trained Transformers image-classification head on a CXR.

    Use a checkpoint with both a compatible image processor and a trained
    classification head. Encoder-only RAD-DINO or CXFormer checkpoints do not
    provide disease predictions until a task-specific head is trained. Scores
    are model outputs, not calibrated clinical probabilities.

    Returns:
        Scores keyed by checkpoint labels. Repeated label names include their
        class IDs to distinguish separate classes.

    See Also:
        https://huggingface.co/docs/transformers/tasks/image_classification
    """
    import torch

    model, processor = get_hf_model_and_processor(model_id)
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
