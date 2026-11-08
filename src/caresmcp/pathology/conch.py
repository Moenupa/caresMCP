"""Pathology image and text embeddings and similarity scores backed by CONCH."""

from functools import lru_cache, partial
from typing import Annotated, Any, Literal

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.imageio import decode_as
from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TASK_EMBED, TASK_RET

NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]

MODEL_ID = "MahmoodLab/conch"
PATHOLOGY_TILE = Field(
    description="Pathology tile or region-of-interest image."
)
NONEMPTY_TEXTS = Field(
    min_length=1,
    description="Pathology descriptions, each containing non-empty text.",
)
CONCH_EMBEDDING_SPACE = Field(
    description="'projected' for the shared image-text space; 'visual' for image features."
)
NORMALIZE_EMBEDDING = Field(description="Whether to return unit-length embeddings.")


@lru_cache(maxsize=1)
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


def _encode_texts(
    model: Any,
    tokenize: Any,
    texts: list[str],
    *,
    normalize: bool = True,
) -> Any:
    """Encode text in CONCH's shared image-text space."""
    return model.encode_text(tokenize(texts).to(DEVICE), normalize=normalize)


def _encode_image(
    model: Any,
    preprocess: Any,
    image: bytes,
    *,
    projected: bool = True,
    normalize: bool = True,
) -> Any:
    """Prepare and encode a single pathology image."""
    import torch

    batch = (
        preprocess(decode_as(image, mode="RGB"))
        .unsqueeze(0)
        .to(DEVICE, dtype=torch.float32)
    )
    return model.encode_image(
        batch,
        proj_contrast=projected,
        normalize=normalize,
    )


@PATHOLOGY.register("conch_image_embed")
@TASK_EMBED.register("conch_image_embed")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_image_embed(
    image: Annotated[bytes, PATHOLOGY_TILE],
    embedding_space: Annotated[
        Literal["projected", "visual"], CONCH_EMBEDDING_SPACE
    ] = "projected",
    normalize: Annotated[bool, NORMALIZE_EMBEDDING] = True,
) -> dict[str, Any]:
    """Encode a pathology image as a numeric feature vector.

    See Also:
        https://huggingface.co/MahmoodLab/CONCH
        https://github.com/Mahmoodlab/CONCH
    """
    import torch

    model, preprocess, _ = _load_conch()
    with torch.inference_mode():
        embedding = _encode_image(
            model,
            preprocess,
            image,
            projected=embedding_space == "projected",
            normalize=normalize,
        )
    values = embedding[0].float().cpu()
    return {
        "model_id": MODEL_ID,
        "embedding_space": embedding_space,
        "normalized": normalize,
        "embedding_size": int(values.numel()),
        "embedding": values.tolist(),
    }


@PATHOLOGY.register("conch_text_embed")
@TASK_EMBED.register("conch_text_embed")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_text_embed(
    texts: Annotated[list[NonemptyText], NONEMPTY_TEXTS],
    normalize: Annotated[bool, NORMALIZE_EMBEDDING] = True,
) -> dict[str, Any]:
    """Encode pathology descriptions as numeric feature vectors.

    These vectors share a space with projected CONCH image embeddings.

    See Also:
        https://huggingface.co/MahmoodLab/CONCH
        https://github.com/Mahmoodlab/CONCH
    """
    import torch

    model, _, tokenize = _load_conch()
    with torch.inference_mode():
        embeddings = _encode_texts(model, tokenize, texts, normalize=normalize)
    values = embeddings.float().cpu().tolist()
    return {
        "model_id": MODEL_ID,
        "normalized": normalize,
        "embedding_size": len(values[0]),
        "embeddings": [
            {"text": text, "embedding": embedding}
            for text, embedding in zip(texts, values, strict=True)
        ],
    }


@PATHOLOGY.register("conch_score_candidate_texts")
@TASK_RET.register("conch_score_candidate_texts")
@requires(["torch", "conch"], extra="pathology-conch")
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
        image_embedding = _encode_image(model, preprocess, image)
        text_embeddings = _encode_texts(model, tokenize, candidate_texts)
        scores = (text_embeddings @ image_embedding[0]).float().cpu().tolist()
    return {
        "model_id": MODEL_ID,
        "score_type": "cosine_similarity",
        "scores": [
            {"text": text, "similarity_score": score}
            for text, score in zip(candidate_texts, scores, strict=True)
        ],
    }
