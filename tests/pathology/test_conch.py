"""Real CONCH inference tests using pathology image URLs."""

import json
import math
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("conch")

from caresmcp.pathology import conch

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

DESCRIPTIONS = [
    "a histopathology image of benign breast fibromatosis",
    "a histopathology image of invasive breast carcinoma",
]
CLASS_NAMES = ["benign breast fibromatosis", "invasive breast carcinoma"]


@pytest.fixture(scope="module")
def projected_tile_embeddings(
    pathology_images: list[bytes],
) -> list[dict[str, Any]]:
    """Extract real projected embeddings for all pathology sample tiles."""
    return [
        conch.conch_image_embed(image, embedding_space="projected")
        for image in pathology_images
    ]


@pytest.fixture(scope="module")
def text_embeddings() -> dict[str, Any]:
    """Encode real pathology descriptions once for all tests."""
    return conch.conch_text_embed(DESCRIPTIONS)


def _assert_unit_vector(vector: list[float]) -> None:
    assert vector
    assert all(math.isfinite(value) for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(
        1.0, rel=1e-4
    )


def _assert_json_safe(result: dict[str, Any]) -> None:
    json.dumps(result, allow_nan=False)


def test_conch_embeddings_are_normalized(
    pathology_image: bytes,
    projected_tile_embeddings: list[dict[str, Any]],
    text_embeddings: dict[str, Any],
) -> None:
    """Extract normalized image features and matching text embeddings."""
    projected = projected_tile_embeddings[0]
    visual = conch.conch_image_embed(pathology_image, embedding_space="visual")

    for result in (projected, visual):
        assert result["embedding_size"] > 0
        _assert_unit_vector(result["embedding"])
        _assert_json_safe(result)

    assert projected["embedding_space"] == "projected"
    assert visual["embedding_space"] == "visual"
    assert text_embeddings["embedding_size"] > 0
    assert [item["text"] for item in text_embeddings["embeddings"]] == DESCRIPTIONS
    for item in text_embeddings["embeddings"]:
        _assert_unit_vector(item["embedding"])
    _assert_json_safe(text_embeddings)


def test_conch_image_text_similarity(
    pathology_image: bytes,
    projected_tile_embeddings: list[dict[str, Any]],
    text_embeddings: dict[str, Any],
) -> None:
    """Score raw descriptions and zero-shot classes against image features."""
    # pathology_image is SAMPLES[1]; compare against the same tile's embedding
    image_vector = projected_tile_embeddings[1]["embedding"]
    scores = conch.conch_score_candidate_texts(pathology_image, DESCRIPTIONS)
    classification = conch.conch_zero_shot_classify(
        pathology_image,
        class_names=CLASS_NAMES,
        prompt_templates=["a histopathology image of {class_name}"],
        class_synonyms=[["benign fibrous breast lesion"], ["breast cancer"]],
    )

    assert scores["score_type"] == "cosine_similarity"
    assert [item["text"] for item in scores["scores"]] == DESCRIPTIONS
    for score, text_embedding in zip(
        scores["scores"], text_embeddings["embeddings"], strict=True
    ):
        expected = sum(
            image_value * text_value
            for image_value, text_value in zip(
                image_vector, text_embedding["embedding"], strict=True
            )
        )
        assert score["similarity_score"] == pytest.approx(expected, rel=1e-2)

    assert classification["predicted_class"] in CLASS_NAMES
    assert [
        item["class_name"] for item in classification["class_scores"]
    ] == CLASS_NAMES
    assert all(
        math.isfinite(item["cosine_similarity"])
        for item in classification["class_scores"]
    )
    _assert_json_safe(scores)
    _assert_json_safe(classification)


def test_conch_tile_bag_operations(
    projected_tile_embeddings: list[dict[str, Any]],
    pathology_images: list[bytes],
) -> None:
    """Classify pooled tile embeddings and rank them for text retrieval."""
    image_vectors = [result["embedding"] for result in projected_tile_embeddings]
    image_ids = [f"tile_{index}" for index in range(len(pathology_images))]
    query = DESCRIPTIONS[0]
    pooled = conch.conch_mizero_classify_tiles(
        image_vectors,
        class_names=CLASS_NAMES,
        prompt_templates=["a histopathology image of {class_name}"],
        top_j=[1, 2, 10],
    )
    retrieval = conch.conch_retrieve_images_by_text(
        query,
        image_ids,
        image_vectors,
        top_k=2,
    )

    assert pooled["tile_count"] == len(projected_tile_embeddings)
    assert [item["tiles_pooled"] for item in pooled["pooled_scores"]] == [1, 2, 3]
    assert all(
        item["predicted_class"] in CLASS_NAMES for item in pooled["pooled_scores"]
    )
    matches = retrieval["matches"]
    assert retrieval["query_text"] == query
    assert len(matches) == 2
    assert len({match["image_id"] for match in matches}) == 2
    assert matches[0]["cosine_similarity"] >= matches[1]["cosine_similarity"]
    _assert_json_safe(pooled)
    _assert_json_safe(retrieval)
