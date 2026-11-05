"""Real CONCH inference tests using pathology image URLs."""

import json
import math
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("conch")

from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.pathology import conch

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


@pytest.fixture(scope="module")
def projected_tile_embeddings(
    pathology_images: list[bytes],
) -> list[dict[str, Any]]:
    """Extract real projected embeddings for all pathology sample tiles."""
    results = [
        conch.conch_image_embed(image, embedding_space="projected")
        for image in pathology_images
    ]
    for index, result in enumerate(results):
        save_results(result, CACHE_ROOT / "conch_image_embed" / f"tile_{index}")
    return results


def _assert_unit_vector(vector: list[float]) -> None:
    assert vector
    assert all(math.isfinite(value) for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(
        1.0, rel=1e-4
    )


def _assert_json_safe(result: dict[str, Any]) -> None:
    json.dumps(result, allow_nan=False)


def test_conch_extract_embedding_returns_projected_and_visual_features(
    pathology_image: bytes,
    projected_tile_embeddings: list[dict[str, Any]],
) -> None:
    """Extract finite, normalized image features in both CONCH feature spaces."""
    projected = projected_tile_embeddings[0]
    visual = conch.conch_image_embed(pathology_image, embedding_space="visual")

    assert projected["model_id"] == conch.MODEL_ID
    assert projected["embedding_space"] == "projected"
    assert visual["embedding_space"] == "visual"
    assert projected["embedding_size"] > 0
    assert visual["embedding_size"] > 0
    _assert_unit_vector(projected["embedding"])
    _assert_unit_vector(visual["embedding"])
    _assert_json_safe(projected)
    _assert_json_safe(visual)
    save_results(projected, CACHE_ROOT / "conch_image_embed_projected")
    save_results(visual, CACHE_ROOT / "conch_image_embed_visual")


def test_conch_extract_text_embeddings_returns_normalized_vectors() -> None:
    """Encode real pathology descriptions in CONCH's projected text space."""
    descriptions = [
        "a histopathology image of benign breast fibromatosis",
        "a histopathology image of invasive breast carcinoma",
    ]
    result = conch.conch_text_embed(descriptions)

    assert result["model_id"] == conch.MODEL_ID
    assert result["embedding_size"] > 0
    assert [item["text"] for item in result["embeddings"]] == descriptions
    for item in result["embeddings"]:
        _assert_unit_vector(item["embedding"])
    _assert_json_safe(result)
    save_results(result, CACHE_ROOT / "conch_text_embed")


def test_conch_score_candidate_texts_matches_real_embedding_cosines(
    pathology_image: bytes,
    projected_tile_embeddings: list[dict[str, Any]],
) -> None:
    """Score descriptions against a tile and verify against real embeddings."""
    descriptions = [
        "a histopathology image of benign breast fibromatosis",
        "a histopathology image of invasive breast carcinoma",
    ]
    # pathology_image is SAMPLES[1]; compare against the same tile's embedding
    image_vector = projected_tile_embeddings[1]["embedding"]
    text_result = conch.conch_text_embed(descriptions)
    score_result = conch.conch_score_candidate_texts(pathology_image, descriptions)

    assert score_result["score_type"] == "cosine_similarity"
    assert [item["text"] for item in score_result["scores"]] == descriptions
    for score, text_embedding in zip(
        score_result["scores"], text_result["embeddings"], strict=True
    ):
        expected = sum(
            image_value * text_value
            for image_value, text_value in zip(
                image_vector, text_embedding["embedding"], strict=True
            )
        )
        assert score["similarity_score"] == pytest.approx(expected, rel=1e-2)
    _assert_json_safe(score_result)
    save_results(score_result, CACHE_ROOT / "conch_score_candidate_texts")


def test_conch_zero_shot_classify_returns_ranked_classes(
    pathology_image: bytes,
) -> None:
    """Rank real pathology descriptions with prompt and synonym ensembles."""
    class_names = ["benign breast fibromatosis", "invasive breast carcinoma"]
    result = conch.conch_zero_shot_classify(
        pathology_image,
        class_names=class_names,
        prompt_templates=["a histopathology image of {class_name}"],
        class_synonyms=[["benign fibrous breast lesion"], ["breast cancer"]],
    )

    assert result["model_id"] == conch.MODEL_ID
    assert result["predicted_class"] in class_names
    assert [item["class_name"] for item in result["class_scores"]] == class_names
    assert all(
        math.isfinite(item["cosine_similarity"]) for item in result["class_scores"]
    )
    _assert_json_safe(result)
    save_results(result, CACHE_ROOT / "conch_zero_shot_classify")


def test_conch_mizero_classifies_real_tile_embeddings(
    projected_tile_embeddings: list[dict[str, Any]],
) -> None:
    """Pool real image features and CONCH text scores across a tile bag."""
    class_names = ["benign breast fibromatosis", "invasive breast carcinoma"]
    result = conch.conch_mizero_classify_tiles(
        [item["embedding"] for item in projected_tile_embeddings],
        class_names=class_names,
        prompt_templates=["a histopathology image of {class_name}"],
        top_j=[1, 2, 10],
    )

    assert result["model_id"] == conch.MODEL_ID
    assert result["tile_count"] == len(projected_tile_embeddings)
    assert [item["tiles_pooled"] for item in result["pooled_scores"]] == [1, 2, 3]
    assert all(
        item["predicted_class"] in class_names for item in result["pooled_scores"]
    )
    _assert_json_safe(result)
    save_results(result, CACHE_ROOT / "conch_mizero_classify_tiles")


def test_conch_retrieves_real_image_embeddings_from_text(
    pathology_images: list[bytes],
    projected_tile_embeddings: list[dict[str, Any]],
) -> None:
    """Rank sample tile embeddings for a real CONCH text query."""
    image_ids = [f"tile_{index}" for index in range(len(pathology_images))]
    image_vectors = [item["embedding"] for item in projected_tile_embeddings]
    query = "a histopathology image of benign breast fibromatosis"
    result = conch.conch_retrieve_images_by_text(
        query,
        image_ids,
        image_vectors,
        top_k=2,
    )

    matches = result["matches"]
    assert result["model_id"] == conch.MODEL_ID
    assert result["query_text"] == query
    assert len(matches) == 2
    assert len({match["image_id"] for match in matches}) == 2
    assert matches[0]["cosine_similarity"] >= matches[1]["cosine_similarity"]
    _assert_json_safe(result)
    save_results(result, CACHE_ROOT / "conch_retrieve_images_by_text")
