"""Image, text, and zero-shot pathology tools backed by CONCH."""

from functools import lru_cache, partial
from importlib import import_module
from typing import Any, Literal

from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TAG_PATHOLOGY
from caresmcp.tools.image_processing import decode_rgb
from caresmcp.util.moduleloader import requires

MODEL_ID = "MahmoodLab/conch"
DEFAULT_PROMPT_TEMPLATES = ["a histopathology image of {class_name}"]
DEFAULT_TOP_J = [1, 5, 10, 50, 100]


@lru_cache(maxsize=1)
def _load_conch() -> tuple[Any, Any, Any]:
    """Load and cache CONCH, its image transform, and a bound text tokenizer."""
    conch_model_module = import_module("conch.open_clip_custom")

    model, preprocess = conch_model_module.create_model_from_pretrained(
        "conch_ViT-B-16",
        "hf_hub:MahmoodLab/conch",
    )
    model = model.to(DEVICE)
    model.eval()
    tokenizer = conch_model_module.get_tokenizer()
    tokenize = partial(conch_model_module.tokenize, tokenizer)
    return model, preprocess, tokenize


def _prepare_conch_image(image: bytes, preprocess: Any) -> Any:
    """Decode and transform image bytes for CONCH."""
    import torch

    try:
        decoded = decode_rgb(image)
    except (OSError, ValueError) as exc:
        raise ValueError("image must contain a readable pathology tile") from exc
    return preprocess(decoded).unsqueeze(0).to(DEVICE, dtype=torch.float32)


def _validate_texts(texts: list[str], name: str) -> None:
    """Validate a non-empty list of non-empty strings."""
    if not isinstance(texts, list) or not texts:
        raise ValueError(f"{name} must contain at least one description")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(f"{name} must not contain empty descriptions")


def _encode_texts(
    model: Any,
    tokenize: Any,
    texts: list[str],
    normalize: bool,
) -> Any:
    """Encode validated text strings and check the returned feature matrix."""
    import torch

    tokens = tokenize(texts).to(DEVICE)
    embeddings = model.encode_text(tokens, normalize=normalize)
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(texts)
        or embeddings.shape[1] == 0
        or not torch.isfinite(embeddings).all()
    ):
        raise RuntimeError("CONCH returned invalid text embeddings")
    return embeddings


def _validate_class_inputs(
    class_names: list[str],
    prompt_templates: list[str] | None,
    class_synonyms: list[list[str]] | None,
) -> tuple[list[str], list[list[str]]]:
    """Validate and expand zero-shot class names into text prompts."""
    _validate_texts(class_names, "class_names")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")

    templates = (
        DEFAULT_PROMPT_TEMPLATES if prompt_templates is None else prompt_templates
    )
    _validate_texts(templates, "prompt_templates")
    if any("{class_name}" not in template for template in templates):
        raise ValueError("each prompt template must include '{class_name}'")

    if class_synonyms is None:
        synonyms = [[] for _ in class_names]
    else:
        if (
            not isinstance(class_synonyms, list)
            or len(class_synonyms) != len(class_names)
            or any(not isinstance(terms, list) for terms in class_synonyms)
        ):
            raise ValueError("class_synonyms must have one list per class_name")
        for terms in class_synonyms:
            if any(not isinstance(term, str) or not term.strip() for term in terms):
                raise ValueError("class_synonyms must not contain empty descriptions")
        synonyms = class_synonyms

    prompts_by_class = [
        [
            template.format(class_name=term)
            for term in [class_name, *terms]
            for template in templates
        ]
        for class_name, terms in zip(class_names, synonyms, strict=True)
    ]
    return templates, prompts_by_class


def _encode_class_embeddings(
    model: Any,
    tokenize: Any,
    prompt_groups: list[list[str]],
) -> Any:
    """Build normalized class vectors by averaging normalized prompt vectors."""
    import torch
    from torch.nn import functional

    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    prompt_embeddings = _encode_texts(model, tokenize, flat_prompts, normalize=True)
    class_embeddings = []
    offset = 0
    for prompt_group in prompt_groups:
        group_embeddings = prompt_embeddings[offset : offset + len(prompt_group)]
        class_embedding = group_embeddings.mean(dim=0)
        class_embedding = functional.normalize(class_embedding, dim=0)
        if not torch.isfinite(class_embedding).all():
            raise RuntimeError("CONCH returned an invalid class embedding")
        class_embeddings.append(class_embedding)
        offset += len(prompt_group)
    return torch.stack(class_embeddings, dim=0)


def _encode_image(
    model: Any,
    pixels: Any,
    *,
    projected: bool,
    normalize: bool,
) -> Any:
    """Encode image pixels and validate one feature vector per input image."""
    import torch

    embeddings = model.encode_image(
        pixels,
        proj_contrast=projected,
        normalize=normalize,
    )
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] != pixels.shape[0]
        or embeddings.shape[1] == 0
        or not torch.isfinite(embeddings).all()
    ):
        raise RuntimeError("CONCH returned invalid image embeddings")
    return embeddings


@MEDAI.register("conch_extract_embedding")
@TAG_PATHOLOGY.register("conch_extract_embedding")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_extract_embedding(
    image: bytes,
    embedding_space: Literal["projected", "visual"] = "projected",
    normalize: bool = True,
) -> dict[str, Any]:
    """Extract a CONCH image embedding from one pathology tile or ROI.

    The projected space is aligned with CONCH text embeddings. The visual space
    is the image encoder representation before its contrastive projection.
    Embeddings are representations, not diagnoses or calibrated probabilities.
    CONCH weights are restricted to non-commercial academic research.

    Args:
        image: Encoded RGB or grayscale pathology tile or region of interest.
        embedding_space: Use the contrastive image-text space or the visual
            representation before the contrastive projection.
        normalize: L2-normalize the returned embedding.

    Returns:
        Model provenance, embedding space, normalization setting, size, and
        embedding values.

    See Also:
        https://huggingface.co/MahmoodLab/CONCH
        https://github.com/Mahmoodlab/CONCH
    """
    import torch

    if embedding_space not in ("projected", "visual"):
        raise ValueError("embedding_space must be 'projected' or 'visual'")
    model, preprocess, _ = _load_conch()
    pixels = _prepare_conch_image(image, preprocess)
    with torch.inference_mode():
        embedding = _encode_image(
            model,
            pixels,
            projected=embedding_space == "projected",
            normalize=normalize,
        )
    values = embedding[0].detach().float().cpu()
    return {
        "model_id": MODEL_ID,
        "embedding_space": embedding_space,
        "normalized": normalize,
        "embedding_size": int(values.numel()),
        "embedding": values.tolist(),
    }


@MEDAI.register("conch_extract_text_embeddings")
@TAG_PATHOLOGY.register("conch_extract_text_embeddings")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_extract_text_embeddings(
    texts: list[str],
    normalize: bool = True,
) -> dict[str, Any]:
    """Extract CONCH text embeddings for pathology descriptions.

    Text embeddings use the same projected space as projected image
    embeddings. The vectors are representations, not diagnoses or calibrated
    probabilities. CONCH weights are restricted to non-commercial academic
    research.

    Args:
        texts: Non-empty pathology descriptions to encode.
        normalize: L2-normalize each returned embedding.

    Returns:
        Model provenance and a JSON-safe embedding for each supplied text.

    See Also:
        https://huggingface.co/MahmoodLab/CONCH
        https://github.com/Mahmoodlab/CONCH
    """
    import torch

    _validate_texts(texts, "texts")
    model, _, tokenize = _load_conch()
    with torch.inference_mode():
        embeddings = _encode_texts(model, tokenize, texts, normalize=normalize)
    values = embeddings.detach().float().cpu().tolist()
    return {
        "model_id": MODEL_ID,
        "normalized": normalize,
        "embedding_size": len(values[0]),
        "embeddings": [
            {"text": text, "embedding": embedding}
            for text, embedding in zip(texts, values, strict=True)
        ],
    }


@MEDAI.register("conch_score_candidate_texts")
@TAG_PATHOLOGY.register("conch_score_candidate_texts")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_score_candidate_texts(
    image: bytes,
    candidate_texts: list[str],
) -> dict[str, Any]:
    """Compare a pathology tile with candidate descriptions using CONCH.

    Scores are cosine similarities, not probabilities or diagnostic confidence.
    CONCH weights are restricted to non-commercial academic research.

    Args:
        image: Encoded RGB or grayscale pathology tile or region of interest.
        candidate_texts: Non-empty candidate descriptions to compare with the
            image. Texts should describe visual content rather than request a
            free-form response.

    Returns:
        The model identifier and candidate descriptions in supplied order,
        each with a cosine similarity score.

    See Also:
        https://huggingface.co/MahmoodLab/CONCH
        https://github.com/Mahmoodlab/CONCH
    """
    import torch

    _validate_texts(candidate_texts, "candidate_texts")
    model, preprocess, tokenize = _load_conch()
    pixels = _prepare_conch_image(image, preprocess)
    with torch.inference_mode():
        image_embedding = _encode_image(
            model,
            pixels,
            projected=True,
            normalize=True,
        )
        text_embeddings = _encode_texts(
            model,
            tokenize,
            candidate_texts,
            normalize=True,
        )
        if image_embedding.shape[1] != text_embeddings.shape[1]:
            raise RuntimeError("CONCH returned incompatible image and text embeddings")
        scores = (text_embeddings @ image_embedding.transpose(0, 1)).reshape(-1)
    return {
        "model_id": MODEL_ID,
        "score_type": "cosine_similarity",
        "scores": [
            {"text": text, "similarity_score": float(score)}
            for text, score in zip(candidate_texts, scores.cpu().tolist(), strict=True)
        ],
    }


@MEDAI.register("conch_zero_shot_classify")
@TAG_PATHOLOGY.register("conch_zero_shot_classify")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_zero_shot_classify(
    image: bytes,
    class_names: list[str],
    prompt_templates: list[str] | None = None,
    class_synonyms: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Rank candidate pathology classes using zero-shot prompt ensembles.

    Similarity scores are not probabilities or diagnostic confidence. This
    model is for non-commercial academic research and is not validated for
    clinical decisions.

    Args:
        image: Encoded RGB or grayscale pathology tile or region of interest.
        class_names: Unique class labels to compare against the image.
        prompt_templates: Optional templates containing ``{class_name}``.
            Each template is evaluated for every class and synonym. If omitted,
            a general histopathology prompt is used.
        class_synonyms: Optional synonym list for each class, in class order.

    Returns:
        The highest-scoring class and cosine similarity scores in supplied
        class order.

    See Also:
        https://github.com/Mahmoodlab/CONCH/blob/main/conch/downstream/zeroshot_path.py
        https://huggingface.co/MahmoodLab/CONCH
    """
    import torch

    _, prompt_groups = _validate_class_inputs(
        class_names,
        prompt_templates,
        class_synonyms,
    )
    model, preprocess, tokenize = _load_conch()
    pixels = _prepare_conch_image(image, preprocess)
    with torch.inference_mode():
        image_embedding = _encode_image(
            model,
            pixels,
            projected=True,
            normalize=True,
        )
        class_embeddings = _encode_class_embeddings(model, tokenize, prompt_groups)
        if image_embedding.shape[1] != class_embeddings.shape[1]:
            raise RuntimeError("CONCH returned incompatible image and text embeddings")
        scores = (class_embeddings @ image_embedding[0]).reshape(-1)
    class_scores = [
        {"class_name": class_name, "cosine_similarity": float(score)}
        for class_name, score in zip(class_names, scores.cpu().tolist(), strict=True)
    ]
    predicted_class = max(class_scores, key=lambda item: item["cosine_similarity"])
    return {
        "model_id": MODEL_ID,
        "predicted_class": predicted_class["class_name"],
        "score_type": "cosine_similarity",
        "class_scores": class_scores,
    }


@MEDAI.register("conch_mizero_classify_tiles")
@TAG_PATHOLOGY.register("conch_mizero_classify_tiles")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_mizero_classify_tiles(
    tile_embeddings: list[list[float]],
    class_names: list[str],
    prompt_templates: list[str] | None = None,
    class_synonyms: list[list[str]] | None = None,
    top_j: list[int] | None = None,
) -> dict[str, Any]:
    """Classify a bag of pathology tiles with CONCH MI-Zero pooling.

    Tile embeddings should come from one whole-slide image or ROI collection
    and use CONCH's projected image-text space. This tool pools the highest
    tile-level class similarities for each requested top-j value. Scores are
    not probabilities or diagnostic confidence.

    Args:
        tile_embeddings: Non-empty projected CONCH image embeddings, one per
            tile. These can be obtained with ``conch_extract_embedding``.
        class_names: Unique class labels to compare against the tiles.
        prompt_templates: Optional templates containing ``{class_name}``.
        class_synonyms: Optional synonym list for each class, in class order.
        top_j: Positive tile counts to pool. Values larger than the tile count
            use all available tiles. Defaults to 1, 5, 10, 50, and 100.

    Returns:
        Model provenance, tile count, and pooled class scores for each top-j.

    See Also:
        https://github.com/Mahmoodlab/CONCH/blob/main/conch/downstream/zeroshot_path.py
    """
    import torch

    if not isinstance(tile_embeddings, list) or not tile_embeddings:
        raise ValueError("tile_embeddings must contain at least one vector")
    _, prompt_groups = _validate_class_inputs(
        class_names,
        prompt_templates,
        class_synonyms,
    )
    selected_top_j = DEFAULT_TOP_J if top_j is None else top_j
    if (
        not isinstance(selected_top_j, list)
        or not selected_top_j
        or any(not isinstance(value, int) or value <= 0 for value in selected_top_j)
        or len(set(selected_top_j)) != len(selected_top_j)
    ):
        raise ValueError("top_j must contain unique positive integers")

    try:
        image_embeddings = torch.tensor(tile_embeddings, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("tile_embeddings must be rectangular numeric vectors") from exc
    if image_embeddings.ndim != 2 or not torch.isfinite(image_embeddings).all():
        raise ValueError("tile_embeddings must be finite vectors with equal dimensions")
    if torch.any(torch.linalg.vector_norm(image_embeddings, dim=1) == 0):
        raise ValueError("tile_embeddings must have non-zero norms")

    model, _, tokenize = _load_conch()
    with torch.inference_mode():
        class_embeddings = _encode_class_embeddings(model, tokenize, prompt_groups)
        if image_embeddings.shape[1] != class_embeddings.shape[1]:
            raise ValueError("tile_embeddings must use CONCH's projected image space")
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)
        tile_scores = image_embeddings @ class_embeddings.transpose(0, 1)
        pooled_results = []
        for requested_j in selected_top_j:
            count = min(requested_j, len(tile_embeddings))
            pooled_scores = tile_scores.topk(count, dim=0).values.mean(dim=0)
            class_scores = [
                {"class_name": class_name, "cosine_similarity": float(score)}
                for class_name, score in zip(
                    class_names,
                    pooled_scores.cpu().tolist(),
                    strict=True,
                )
            ]
            predicted_class = max(
                class_scores,
                key=lambda item: item["cosine_similarity"],
            )
            pooled_results.append(
                {
                    "top_j": requested_j,
                    "tiles_pooled": count,
                    "predicted_class": predicted_class["class_name"],
                    "class_scores": class_scores,
                }
            )
    return {
        "model_id": MODEL_ID,
        "tile_count": len(tile_embeddings),
        "score_type": "mean_top_j_cosine_similarity",
        "pooled_scores": pooled_results,
    }


@MEDAI.register("conch_retrieve_images_by_text")
@TAG_PATHOLOGY.register("conch_retrieve_images_by_text")
@requires(["torch", "conch"], extra="pathology-conch")
def conch_retrieve_images_by_text(
    query_text: str,
    candidate_image_ids: list[str],
    candidate_image_embeddings: list[list[float]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve projected CONCH image embeddings that best match a text query.

    Candidate vectors must come from ``conch_extract_embedding`` with the
    ``projected`` embedding space. This enables text-to-image retrieval without
    resending the original images.

    Args:
        query_text: Non-empty pathology description to search for.
        candidate_image_ids: Unique identifiers paired with candidate vectors.
        candidate_image_embeddings: Projected image embeddings paired with the
            identifiers, in the same order.
        top_k: Maximum number of matching image identifiers to return.

    Returns:
        The highest cosine similarity matches in descending order.

    See Also:
        https://huggingface.co/MahmoodLab/CONCH
    """
    import torch
    from torch.nn import functional

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must not be empty")
    count = len(candidate_image_ids)
    if count == 0 or len(candidate_image_embeddings) != count:
        raise ValueError(
            "candidate IDs and embeddings must have equal non-zero lengths"
        )
    if any(
        not isinstance(image_id, str) or not image_id.strip()
        for image_id in candidate_image_ids
    ):
        raise ValueError("candidate_image_ids must not contain empty identifiers")
    if len(set(candidate_image_ids)) != count:
        raise ValueError("candidate_image_ids must be unique")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be positive")

    model, _, tokenize = _load_conch()
    with torch.inference_mode():
        query_embedding = _encode_texts(model, tokenize, [query_text], normalize=True)[
            0
        ]
    try:
        image_embeddings = torch.tensor(candidate_image_embeddings, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "candidate_image_embeddings must be rectangular numeric vectors"
        ) from exc
    if (
        image_embeddings.ndim != 2
        or image_embeddings.shape[0] != count
        or image_embeddings.shape[1] != query_embedding.numel()
        or not torch.isfinite(image_embeddings).all()
    ):
        raise ValueError(
            "candidate embeddings must be finite vectors matching CONCH dimensions"
        )
    if torch.any(torch.linalg.vector_norm(image_embeddings, dim=1) == 0):
        raise ValueError("candidate embeddings must have non-zero norms")

    scores = functional.normalize(image_embeddings, dim=1) @ query_embedding.cpu()
    match_count = min(top_k, count)
    best_scores, best_indices = torch.topk(scores, k=match_count)
    matches = [
        {"image_id": candidate_image_ids[index], "cosine_similarity": float(score)}
        for index, score in zip(
            best_indices.tolist(),
            best_scores.tolist(),
            strict=True,
        )
    ]
    return {"model_id": MODEL_ID, "query_text": query_text, "matches": matches}
