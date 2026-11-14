"""Pathology image and text similarity scores backed by CONCH."""

from functools import partial
from typing import Annotated, Any

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as
from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TASK_RET

NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]

MODEL_ID = "MahmoodLab/conch"
PATHOLOGY_TILE = Field(description="Pathology tile or region-of-interest image.")
NONEMPTY_TEXTS = Field(
    min_length=1,
    description="Pathology descriptions, each containing non-empty text.",
)


def _load_conch() -> tuple[Any, Any, Any]:
    """Load and cache CONCH, its image transform, and a bound text tokenizer."""
    from conch.open_clip_custom import (
        create_model_from_pretrained,
        get_tokenizer,
        tokenize,
    )

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        f"hf_hub:{MODEL_ID}",
    )
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, partial(tokenize, get_tokenizer())


@PATHOLOGY.register("conch_score_candidate_texts")
@TASK_RET.register("conch_score_candidate_texts")
@requires(["torch", "conch"], extra="pathology")
def conch_score_candidate_texts(
    image: Annotated[bytes, PATHOLOGY_TILE],
    candidate_texts: Annotated[list[NonemptyText], NONEMPTY_TEXTS],
) -> dict[str, Any]:
    """Score how closely each candidate description matches a pathology image.

    See Also:
        https://huggingface.co/MahmoodLab/CONCH
        https://github.com/Mahmoodlab/CONCH
    """
    import torch

    model, preprocess, tokenize = _load_conch()
    with torch.inference_mode():
        batch = (
            preprocess(decode_as(image, mode="RGB"))
            .unsqueeze(0)
            .to(DEVICE, dtype=torch.float32)
        )
        image_embedding = model.encode_image(batch, proj_contrast=True, normalize=True)
        text_embeddings = model.encode_text(
            tokenize(candidate_texts).to(DEVICE), normalize=True
        )
        scores = (text_embeddings @ image_embedding[0]).float().cpu().tolist()
    return {
        "model_id": MODEL_ID,
        "score_type": "cosine_similarity",
        "scores": [
            {"text": text, "similarity_score": score}
            for text, score in zip(candidate_texts, scores, strict=True)
        ],
    }
