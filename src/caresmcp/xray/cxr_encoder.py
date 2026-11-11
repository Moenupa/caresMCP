"""RAD-DINO feature extraction for the unified X-ray embedding interface."""

from typing import Any, Literal

from caresmcp.api.imageio import decode_as
from caresmcp.api.models import HFModelType, get_hf_model_and_processor
from caresmcp.constants import DEVICE

CXRModel = Literal["microsoft/rad-dino"]


def _extract_cxr_embedding(
    image: bytes,
    model_id: CXRModel,
    pooling: Literal["cls", "mean"] = "cls",
) -> dict[str, Any]:
    """Return a chest radiograph feature vector with its encoder and pooling."""
    import torch

    model, processor = get_hf_model_and_processor(
        model_id,
        model_type=HFModelType.IMAGE_ENCODER,
    )

    inputs = processor(images=decode_as(image, mode="RGB"), return_tensors="pt")
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)

    tokens = getattr(output, "last_hidden_state", None)
    if tokens is None or tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError(
            "model must return last_hidden_state with shape [batch, tokens, features]"
        )
    if pooling == "cls":
        vector = tokens[:, 0, :]
    else:
        # DINOv2 variants may insert register tokens between CLS and patches.
        first_patch = 1 + getattr(model.config, "num_register_tokens", 0)
        if tokens.shape[1] <= first_patch:
            raise ValueError("model must return image patch tokens for mean pooling")
        vector = tokens[:, first_patch:, :].mean(dim=1)
    if vector.shape[-1] == 0 or not torch.isfinite(vector).all():
        raise ValueError("model must return a nonempty finite embedding")

    embedding = vector[0].detach().float().cpu().tolist()
    return {
        "model_id": model_id,
        "pooling": pooling,
        "representation": "cls_token" if pooling == "cls" else "mean_patch_tokens",
        "embedding": [float(value) for value in embedding],
    }
