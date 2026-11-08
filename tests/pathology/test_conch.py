"""Real CONCH inference tests using pathology image URLs."""

import json
import math
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("conch")

from caresmcp.api.decorators import save_returns
from caresmcp.pathology import conch

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

DESCRIPTIONS = [
    "a histopathology image of benign breast fibromatosis",
    "a histopathology image of invasive breast carcinoma",
]


@pytest.fixture(scope="module")
def projected_embedding(pathology_image: bytes):
    """Extract projected embeddings for the pathology sample tile."""
    return conch.conch_image_embed(pathology_image, embedding_space="projected")


@pytest.fixture(scope="module")
def text_embeddings():
    """Encode real pathology descriptions once for all tests."""
    return conch.conch_text_embed(DESCRIPTIONS)


def _assert_unit_vector(vector: list[float]):
    assert vector
    assert all(math.isfinite(value) for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(
        1.0, rel=1e-4
    )


def _assert_json_safe(result: dict[str, Any]):
    json.dumps(result, allow_nan=False)


@save_returns("conch", unpack=True)
def test_conch_embeddings_are_normalized(
    pathology_image: bytes,
    projected_embedding: dict[str, Any],
    text_embeddings: dict[str, Any],
):
    """Extract normalized image features and matching text embeddings."""
    projected = projected_embedding
    visual = conch.conch_image_embed(pathology_image, embedding_space="visual")

    for result in (projected, visual):
        assert result["embedding_size"] > 0
        assert len(result["embedding"]) == result["embedding_size"]
        _assert_unit_vector(result["embedding"])
        _assert_json_safe(result)

    assert projected["embedding_space"] == "projected"
    assert visual["embedding_space"] == "visual"
    assert text_embeddings["embedding_size"] > 0
    assert [item["text"] for item in text_embeddings["embeddings"]] == DESCRIPTIONS
    for item in text_embeddings["embeddings"]:
        assert len(item["embedding"]) == projected["embedding_size"]
        _assert_unit_vector(item["embedding"])
    _assert_json_safe(text_embeddings)
    return {"projected": projected, "visual": visual, "text": text_embeddings}


@save_returns("conch", unpack=True)
def test_conch_image_text_similarity(
    pathology_image: bytes,
    projected_embedding: dict[str, Any],
    text_embeddings: dict[str, Any],
):
    """Score every candidate description in input order against image features."""
    image_vector = projected_embedding["embedding"]
    scores = conch.conch_score_candidate_texts(pathology_image, DESCRIPTIONS)

    assert len(scores["scores"]) == len(DESCRIPTIONS)
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

    _assert_json_safe(scores)
    return {"scores": scores}
