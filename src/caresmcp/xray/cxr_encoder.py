"""Chest radiograph image encoder and report retrieval tools."""

from typing import Annotated, Any, Literal

from pydantic import Field

from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_CLIP, TASK_EMBED, XRAY
from caresmcp.tools.image_processing import decode_rgb
from caresmcp.util.moduleloader import requires
from caresmcp.util.transformers import (
    HFModelType,
    _get_hf_model_and_processor,
)

CXRModel = Literal[
    "microsoft/rad-dino",
    "m42-health/CXformer-small",
    "m42-health/CXformer-base",
]

CHEST_RADIOGRAPH_FIELD = Field(description="Chest radiograph image")
CXR_MODEL_FIELD = Field(description="Supported Hugging Face chest X-ray encoder")
CXR_POOLING_FIELD = Field(
    description="Use the CLS token or mean-pool the non-CLS tokens"
)
REFERENCE_REPORTS_FIELD = Field(
    min_length=1,
    description="Radiology reports paired with the reference images",
)
REFERENCE_EMBEDDINGS_FIELD = Field(
    min_length=1,
    description="Reference image vectors from the same encoder and pooling method",
)
TOP_K_FIELD = Field(ge=1, description="Maximum number of report matches to return")


def _extract_cxr_embedding(
    image: bytes,
    model_id: CXRModel,
    pooling: Literal["cls", "mean"] = "cls",
) -> dict[str, Any]:
    """Extract an image embedding using a supported chest radiograph encoder."""
    import torch

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
    reference_reports: list[str],
    reference_image_embeddings: list[list[float]],
    model_id: CXRModel,
    reference_pooling: Literal["cls", "mean"] = "cls",
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve paired reports by image embedding similarity."""
    import torch
    from torch.nn import functional

    count = len(reference_reports)
    if len(reference_image_embeddings) != count:
        raise ValueError(
            "reference reports and embeddings must have equal lengths"
        )

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


@XRAY.register("cxr_encoder_image_embed")
@TASK_EMBED.register("cxr_encoder_image_embed")
@requires(["torch", "transformers"], extra="hf")
def cxr_encoder_image_embed(
    image: Annotated[bytes, CHEST_RADIOGRAPH_FIELD],
    model_id: Annotated[CXRModel, CXR_MODEL_FIELD] = "microsoft/rad-dino",
    pooling: Annotated[Literal["cls", "mean"], CXR_POOLING_FIELD] = "cls",
) -> dict[str, Any]:
    """Create a visual embedding for a chest radiograph.

    The embedding represents image features and does not provide disease
    labels.

    Returns:
        The model ID, pooling method, and embedding vector.

    See Also:
        https://huggingface.co/microsoft/rad-dino
        https://huggingface.co/m42-health/CXformer-small
        https://huggingface.co/m42-health/CXformer-base
    """
    return _extract_cxr_embedding(
        image=image,
        model_id=model_id,
        pooling=pooling,
    )


@XRAY.register("cxr_encoder_retrieve_similar_reports")
@TASK_CLIP.register("cxr_encoder_retrieve_similar_reports")
@requires(["torch", "transformers"], extra="hf")
def cxr_encoder_retrieve_similar_reports(
    image: Annotated[bytes, CHEST_RADIOGRAPH_FIELD],
    reference_reports: Annotated[list[str], REFERENCE_REPORTS_FIELD],
    reference_image_embeddings: Annotated[
        list[list[float]], REFERENCE_EMBEDDINGS_FIELD
    ],
    model_id: Annotated[CXRModel, CXR_MODEL_FIELD] = "microsoft/rad-dino",
    reference_pooling: Annotated[
        Literal["cls", "mean"], CXR_POOLING_FIELD
    ] = "cls",
    top_k: Annotated[int, TOP_K_FIELD] = 5,
) -> dict[str, Any]:
    """Find reports paired with the most similar reference chest radiographs.

    This ranks the supplied reports by cosine similarity between image
    embeddings. It does not generate reports or diagnose findings.

    Returns:
        The model ID, pooling method, and matching reports ordered by cosine
        similarity.

    See Also:
        https://huggingface.co/microsoft/rad-dino
        https://huggingface.co/m42-health/CXformer-small
        https://huggingface.co/m42-health/CXformer-base
    """
    return _retrieve_cxr_reports(
        image=image,
        reference_reports=reference_reports,
        reference_image_embeddings=reference_image_embeddings,
        model_id=model_id,
        reference_pooling=reference_pooling,
        top_k=top_k,
    )
