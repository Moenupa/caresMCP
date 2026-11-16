"""PRISM pathology tile analysis for academic research."""

from functools import partial
from typing import Annotated, Any, Literal

from pydantic import Field

from caresmcp.api.decorators import adapt_inputs, requires
from caresmcp.api.io import decode_as
from caresmcp.api.io.tensor import torch_autocast
from caresmcp.api.models import HFMODEL_ID_DESC, get_hf_model
from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TASK_EMB, TASK_RET

from .pathology_encoder import PATHOLOGY_TILE, _extract_pathology_embedding_tensor

MODEL_ID = "paige-ai/Prism"
PrismModel = Annotated[Literal["paige-ai/Prism"], HFMODEL_ID_DESC]
_REQUIRES = requires(
    [
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "einops",
        "environs",
        "sacremoses",
    ],
    extra="pathology",
)


def _prepare_slide(image: Any, model_id: str) -> tuple[Any, Any]:
    """Encode one pathology tile and compute its PRISM representations."""
    import torch

    if model_id != MODEL_ID:
        raise ValueError("model_id must be 'paige-ai/Prism'")
    _, features = _extract_pathology_embedding_tensor(
        image=image,
        model_id="paige-ai/Virchow",
        pooling="cls_mean",
    )
    if features.shape != (2560,) or not torch.isfinite(features).all():
        raise RuntimeError("Virchow returned an invalid 2560-value embedding")

    model: Any = get_hf_model(model_id, trust_remote_code=True)
    with torch.inference_mode(), torch_autocast(dtype=torch.float16):
        representations = model.slide_representations(
            features.reshape(1, 1, 2560).to(DEVICE)
        )
    embedding = representations["image_embedding"]
    if (
        not isinstance(embedding, torch.Tensor)
        or embedding.shape != (1, 1280)
        or not torch.isfinite(embedding).all()
    ):
        raise RuntimeError("PRISM returned an invalid slide embedding")
    return model, representations


@PATHOLOGY.register("prism_slide_embed")
@TASK_EMB.register("prism_slide_embed")
@_REQUIRES
@adapt_inputs(image=partial(decode_as, mode="RGB"))
def prism_slide_embed(
    image: Annotated[bytes, PATHOLOGY_TILE],
    model_id: PrismModel = MODEL_ID,
) -> dict[str, Any]:
    """Encode a pathology tile or ROI as a 1280-dimensional PRISM vector."""
    _, representations = _prepare_slide(image, model_id)
    return {
        "model_id": model_id,
        "tile_encoder": "paige-ai/Virchow",
        "tile_count": 1,
        "embedding_size": 1280,
        "embedding": representations["image_embedding"][0]
        .detach()
        .float()
        .cpu()
        .tolist(),
    }


@PATHOLOGY.register("prism_zero_shot_classify")
@TASK_RET.register("prism_zero_shot_classify")
@_REQUIRES
@adapt_inputs(image=partial(decode_as, mode="RGB"))
def prism_zero_shot_classify(
    image: Annotated[bytes, PATHOLOGY_TILE],
    negative_prompts: Annotated[list[str], Field(min_length=1)],
    positive_prompts: Annotated[list[str], Field(min_length=1)],
    model_id: PrismModel = MODEL_ID,
) -> dict[str, Any]:
    """Compare two groups of pathology descriptions with a tile or ROI using PRISM.

    Returns probabilities for the negative and positive prompt groups, in that
    order. Each group can contain multiple descriptions of the same class.
    """
    import torch

    for prompts in (negative_prompts, positive_prompts):
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("prompt groups must contain non-empty descriptions")
    model, representations = _prepare_slide(image, model_id)
    with torch.inference_mode(), torch_autocast(dtype=torch.float16):
        scores = model.zero_shot(
            representations["image_embedding"],
            neg_prompts=negative_prompts,
            pos_prompts=positive_prompts,
        )
    if (
        not isinstance(scores, torch.Tensor)
        or scores.shape != (1, 2)
        or not torch.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise RuntimeError("PRISM returned invalid zero-shot probabilities")
    negative, positive = scores[0].detach().float().cpu().tolist()
    return {
        "model_id": model_id,
        "tile_count": 1,
        "negative_prompts": negative_prompts,
        "positive_prompts": positive_prompts,
        "probability_negative": negative,
        "probability_positive": positive,
    }
