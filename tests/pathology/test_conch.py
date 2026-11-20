"""Reproduce CONCH's official notebook demo on the repository's own tile.

Every CONCH notebook scores ``docs/roi1.jpg`` with the package calls
``create_model_from_pretrained``, ``encode_image(proj_contrast=True,
normalize=True)``, and ``encode_text`` (see ``notebooks/`` in
https://github.com/Mahmoodlab/CONCH). The model card and README publish only
dataset-level results, such as CRC-100K-Raw balanced accuracy 0.941 with a
K=20 KNN over ROI features; those benchmark splits are not redistributed, so
they cannot be re-measured here. These tests run the notebook computation on
the official tile and require the public tool to reproduce the released
package's similarity scores and demo decisions exactly.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("conch")

from caresmcp.api.decorators import save_returns
from caresmcp.api.io import decode_as
from caresmcp.constants import DEVICE
from caresmcp.pathology import conch

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

# Prompts from notebooks/zeroshot_classification_example_starter.ipynb.
STARTER_PROMPTS = [
    "an H&E image of invasive ductal carcinoma",
    "an H&E image of invasive lobular carcinoma",
]

# Prompts from notebooks/zeroshot_retrieval_example.ipynb.
RETRIEVAL_PROMPTS = [
    "photomicrograph illustrating invasive ductal carcinoma of the breast, H&E stain",
    "a case of invasive lobular carcinoma as visualized using H&E stain",
    "high magnification view of a breast cancer tumor, H&E stain",
    "clear cell renal cell carcinoma",
    "lung adenocarcinoma, H&E stain",
    "IHC stain for CDX2 in a case of metastatic colorectal adenocarcinoma",
    "an image of a cat",
    "High-grade angiosarcoma characterized by solid areas of polygonal and spindled cells as well as necrosis",
    "metastatic tumor to the lymph node, GATA3 staining",
    "epidermis with follicular ostia",
]

PROMPTS = STARTER_PROMPTS + RETRIEVAL_PROMPTS


@pytest.fixture(scope="module")
def official_scores(pathology_image: bytes):
    """Score the demo prompts with the calls used by the official notebooks."""
    import torch
    from conch.open_clip_custom import (
        create_model_from_pretrained,
        get_tokenizer,
        tokenize,
    )

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", f"hf_hub:{conch.MODEL_ID}"
    )
    model = model.eval().to(DEVICE)
    tile = preprocess(decode_as(pathology_image, mode="RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        image_embedding = model.encode_image(tile, proj_contrast=True, normalize=True)
        text_embedding = model.encode_text(
            tokenize(texts=PROMPTS, tokenizer=get_tokenizer()).to(DEVICE),
            normalize=True,
        )
        similarities = (image_embedding @ text_embedding.T).squeeze(0)
    return similarities.float().cpu()


@save_returns("conch", unpack=True)
def test_conch_reproduces_official_notebook(pathology_image: bytes, official_scores):
    """Match the released package and both notebook demo decisions."""
    import torch

    result = conch.conch_score_candidate_texts(pathology_image, PROMPTS)

    assert result["model_id"] == conch.MODEL_ID
    assert result["score_type"] == "cosine_similarity"
    assert [item["text"] for item in result["scores"]] == PROMPTS
    scores = torch.tensor([item["similarity_score"] for item in result["scores"]])
    # Same checkpoint, preprocessing, tokenization, and cosine scores as the
    # official retrieval notebook.
    torch.testing.assert_close(scores, official_scores, rtol=1e-5, atol=1e-6)

    # The starter notebook predicts the maximum softmax score; softmax over
    # candidates is monotone, so the predicted class is the argmax here.
    starter = slice(0, len(STARTER_PROMPTS))
    assert int(scores[starter].argmax()) == int(official_scores[starter].argmax())

    # The retrieval notebook ranks the candidates by cosine similarity.
    retrieval = slice(len(STARTER_PROMPTS), None)
    assert torch.argsort(scores[retrieval], descending=True).tolist() == (
        torch.argsort(official_scores[retrieval], descending=True).tolist()
    )

    return {
        "scores": result,
        "predicted_class": STARTER_PROMPTS[int(scores[starter].argmax())],
    }
