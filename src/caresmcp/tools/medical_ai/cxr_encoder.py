"""Chest radiograph image encoder and report retrieval tools."""

from typing import Any, Literal

from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TAG_XRAY
from caresmcp.tools.medical_ai.transformers import (
    HFModelType,
    _get_hf_model_and_processor,
)
from caresmcp.util.moduleloader import requires

from ..image_processing import decode_rgb

CXRModel = Literal["rad-dino", "cxformer-small", "cxformer-base"]
CXR_ENCODER_MODEL_IDS: dict[CXRModel, str] = {
    "rad-dino": "microsoft/rad-dino",
    "cxformer-small": "m42-health/CXformer-small",
    "cxformer-base": "m42-health/CXformer-base",
}


def _get_model_id(encoder: CXRModel) -> str:
    """Return a supported encoder's Hugging Face model identifier."""
    try:
        return CXR_ENCODER_MODEL_IDS[encoder]
    except KeyError as exc:
        raise ValueError(
            "encoder must be 'rad-dino', 'cxformer-small', or 'cxformer-base'"
        ) from exc


def _extract_cxr_embedding(
    image: bytes,
    model_id: str,
    pooling: Literal["cls", "mean"] = "cls",
) -> dict[str, Any]:
    """Extract a feature vector from a published chest radiograph encoder."""
    import torch

    if pooling not in ("cls", "mean"):
        raise ValueError("pooling must be 'cls' or 'mean'")

    model, processor = _get_hf_model_and_processor(
        model_id,
        model_type=HFModelType.IMAGE_ENCODER,
        trust_remote_code=model_id.startswith("m42-health/"),
    )

    inputs = processor(images=decode_rgb(image), return_tensors="pt")
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)

    tokens = getattr(output, "last_hidden_state", None)
    if tokens is None or tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError(
            "model must return last_hidden_state with shape [batch, tokens, features]"
        )
    if pooling == "cls" or tokens.shape[1] == 1:
        vector = tokens[:, 0, :]
    else:
        vector = tokens[:, 1:, :].mean(dim=1)

    embedding = vector[0].detach().float().cpu().tolist()
    return {
        "model_id": model_id,
        "pooling": pooling,
        "embedding": [float(value) for value in embedding],
    }


def _retrieve_cxr_reports(
    image: bytes,
    reference_ids: list[str],
    reference_reports: list[str],
    reference_image_embeddings: list[list[float]],
    model_id: str,
    reference_pooling: Literal["cls", "mean"] = "cls",
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve reports paired with the nearest reference CXR embeddings.

    This performs image-embedding nearest-neighbor retrieval; it is not the
    separately trained CXR-RePaiR or X-REM pipeline.
    """
    import torch
    from torch.nn import functional

    count = len(reference_ids)
    if count == 0:
        raise ValueError("at least one reference report is required")
    if len(reference_reports) != count or len(reference_image_embeddings) != count:
        raise ValueError(
            "reference IDs, reports, and embeddings must have equal lengths"
        )
    if len(set(reference_ids)) != count:
        raise ValueError("reference_ids must be unique")
    if not 1 <= top_k:
        raise ValueError("top_k must be positive")

    query_result = _extract_cxr_embedding(
        image=image,
        model_id=model_id,
        pooling=reference_pooling,
    )
    query = torch.tensor(query_result["embedding"], dtype=torch.float32)
    feature_size = query.numel()
    if any(len(embedding) != feature_size for embedding in reference_image_embeddings):
        raise ValueError("each reference embedding must match the query feature size")
    references = torch.tensor(reference_image_embeddings, dtype=torch.float32)
    if not torch.isfinite(references).all() or not torch.isfinite(query).all():
        raise ValueError("embeddings must contain only finite values")
    if (
        torch.any(torch.linalg.vector_norm(references, dim=1) == 0)
        or torch.linalg.vector_norm(query) == 0
    ):
        raise ValueError("embeddings must have non-zero norm")
    normalized_query = functional.normalize(query, dim=0)
    normalized_references = functional.normalize(references, dim=1)
    similarities = normalized_references @ normalized_query
    result_count = min(top_k, count)
    scores, indices = torch.topk(similarities, k=result_count)
    matches = [
        {
            "reference_id": reference_ids[index],
            "report": reference_reports[index],
            "cosine_similarity": float(score),
        }
        for index, score in zip(indices.tolist(), scores.tolist(), strict=True)
    ]
    return {
        "model_id": model_id,
        "pooling": reference_pooling,
        "matches": matches,
    }


@MEDAI.register("cxr_encoder_extract_image_embedding")
@TAG_XRAY.register("cxr_encoder_extract_image_embedding")
@requires(["torch", "transformers"], extra="hf")
def cxr_encoder_extract_image_embedding(
    image: bytes,
    encoder: CXRModel = "rad-dino",
    pooling: Literal["cls", "mean"] = "cls",
) -> dict[str, Any]:
    """Extract a general visual feature vector from a chest radiograph.

    The feature vector does not predict disease labels.

    Args:
        image: Chest radiograph image bytes.
        encoder: RAD-DINO, CXFormer small, or CXFormer base checkpoint.
        pooling: Use the first token or the mean of non-CLS tokens.

    Returns:
        Model provenance, pooling method, and a JSON-safe embedding vector.

    See Also:
        https://huggingface.co/microsoft/rad-dino
        https://huggingface.co/m42-health/CXformer-small
        https://huggingface.co/m42-health/CXformer-base
    """
    return _extract_cxr_embedding(
        image=image,
        model_id=_get_model_id(encoder),
        pooling=pooling,
    )


@MEDAI.register("cxr_encoder_retrieve_similar_reports")
@TAG_XRAY.register("cxr_encoder_retrieve_similar_reports")
@requires(["torch", "transformers"], extra="hf")
def cxr_encoder_retrieve_similar_reports(
    image: bytes,
    reference_ids: list[str],
    reference_reports: list[str],
    reference_image_embeddings: list[list[float]],
    encoder: CXRModel = "rad-dino",
    reference_pooling: Literal["cls", "mean"] = "cls",
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve reports paired with nearest CXR image embeddings.

    Each reference report must be paired with an embedding extracted using the
    same encoder and pooling method.

    Args:
        image: Query chest radiograph image bytes.
        reference_ids: Identifiers for indexed studies or images.
        reference_reports: Report text paired with each reference image.
        reference_image_embeddings: Features from the reference images.
        encoder: Checkpoint used for the query and reference embeddings.
        reference_pooling: Pooling method used for the reference features.
        top_k: Maximum number of reports to return.

    Returns:
        Model provenance and the highest cosine-similarity report matches.

    See Also:
        https://huggingface.co/microsoft/rad-dino
        https://huggingface.co/m42-health/CXformer-small
        https://huggingface.co/m42-health/CXformer-base
        https://github.com/rajpurkarlab/CXR-RePaiR
    """
    return _retrieve_cxr_reports(
        image=image,
        reference_ids=reference_ids,
        reference_reports=reference_reports,
        reference_image_embeddings=reference_image_embeddings,
        model_id=_get_model_id(encoder),
        reference_pooling=reference_pooling,
        top_k=top_k,
    )
