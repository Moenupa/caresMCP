import math
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
from caresmcp.api.models.hf_hub import HFModelType
from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TASK_CLS, TASK_EMB


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


@MEDAI.register("hf_image_embed")
@TASK_EMB.register("hf_image_embed")
@requires(["torch", "transformers"], extra="hf")
def hf_image_embed(
    image: Annotated[bytes, Field(description="Image to embed.")],
    model_id: Annotated[
        str, Field(description=HFMODEL_ID_DESC)
    ] = "facebook/dinov2-base",
    pooling: Annotated[
        Literal["cls", "mean"],
        Field(description="First token or mean of non-first tokens."),
    ] = "cls",
) -> dict[str, Any]:
    """Extract image features with a Hugging Face image encoder.

    Use a compatible image encoder and processor. For token outputs, mean
    pooling excludes the first token; for pooled outputs, pooling is supplied
    by the model.
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
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("encoder returned empty or non-finite feature values")
    return {"model_id": model_id, "pooling": pooling_used, "embedding": embedding}
