"""Real CONCH inference tests using pathology image URLs."""

import json
import math

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


@save_returns("conch", unpack=True)
def test_conch_image_text_similarity(pathology_image: bytes):
    """Score every candidate description in input order against the image."""
    scores = conch.conch_score_candidate_texts(pathology_image, DESCRIPTIONS)

    assert scores["model_id"] == conch.MODEL_ID
    assert scores["score_type"] == "cosine_similarity"
    assert [item["text"] for item in scores["scores"]] == DESCRIPTIONS
    for item in scores["scores"]:
        assert math.isfinite(item["similarity_score"])
        assert -1.0001 <= item["similarity_score"] <= 1.0001
    json.dumps(scores, allow_nan=False)
    return {"scores": scores}
