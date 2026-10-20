"""Real CONCH inference tests using bundled pathology tiles."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("conch")

from caresmcp.medical_ai import conch

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "pathology"
SAMPLE_PATHS = tuple(
    sorted(
        path
        for path in SAMPLE_DIRECTORY.glob("breast-fibromatosis-1-*.png")
        if path.is_file()
    )
)


@pytest.fixture(scope="module")
def pathology_tiles() -> list[tuple[str, bytes]]:
    """Return the bundled breast fibromatosis tiles at each magnification."""
    if len(SAMPLE_PATHS) < 3:
        pytest.skip("the three bundled pathology magnifications are required")
    return [(path.stem, path.read_bytes()) for path in SAMPLE_PATHS]


@pytest.fixture(scope="module")
def projected_tile_embeddings(
    pathology_tiles: list[tuple[str, bytes]],
) -> list[dict[str, Any]]:
    """Extract real projected embeddings for all bundled pathology tiles."""
    return [
        conch.conch_image_embed(image, embedding_space="projected")
        for _, image in pathology_tiles
    ]


def _assert_unit_vector(vector: list[float]) -> None:
    assert vector
    assert all(math.isfinite(value) for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(
        1.0, abs=1e-4
    )


def _assert_json_safe(result: dict[str, Any]) -> None:
    json.dumps(result, allow_nan=False)


def test_conch_extract_embedding_returns_projected_and_visual_features(
    pathology_tiles: list[tuple[str, bytes]],
    projected_tile_embeddings: list[dict[str, Any]],
) -> None:
    """Extract finite, normalized image features in both CONCH feature spaces."""
    projected = projected_tile_embeddings[0]
    visual = conch.conch_image_embed(pathology_tiles[0][1], embedding_space="visual")

    assert projected["model_id"] == conch.MODEL_ID
    assert projected["embedding_space"] == "projected"
    assert visual["embedding_space"] == "visual"
    assert projected["embedding_size"] > 0
    assert visual["embedding_size"] > 0
    _assert_unit_vector(projected["embedding"])
    _assert_unit_vector(visual["embedding"])
    _assert_json_safe(projected)
    _assert_json_safe(visual)


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


def test_conch_score_candidate_texts_matches_real_embedding_cosines(
    pathology_tiles: list[tuple[str, bytes]],
    projected_tile_embeddings: list[dict[str, Any]],
) -> None:
    """Score descriptions against a tile and verify against real embeddings."""
    descriptions = [
        "a histopathology image of benign breast fibromatosis",
        "a histopathology image of invasive breast carcinoma",
    ]
    image_vector = projected_tile_embeddings[0]["embedding"]
    text_result = conch.conch_text_embed(descriptions)
    score_result = conch.conch_score_candidate_texts(
        pathology_tiles[0][1], descriptions
    )

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
        assert score["similarity_score"] == pytest.approx(expected, abs=1e-4)
    _assert_json_safe(score_result)


def test_conch_zero_shot_classify_returns_ranked_classes(
    pathology_tiles: list[tuple[str, bytes]],
) -> None:
    """Rank real pathology descriptions with prompt and synonym ensembles."""
    class_names = ["benign breast fibromatosis", "invasive breast carcinoma"]
    result = conch.conch_zero_shot_classify(
        pathology_tiles[0][1],
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


def test_conch_retrieves_real_image_embeddings_from_text(
    pathology_tiles: list[tuple[str, bytes]],
    projected_tile_embeddings: list[dict[str, Any]],
) -> None:
    """Rank bundled tile embeddings for a real CONCH text query."""
    image_ids = [tile_id for tile_id, _ in pathology_tiles]
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
