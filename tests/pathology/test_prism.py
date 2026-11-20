"""Reproduce PRISM's published model-card example and upstream tool outputs.

The PRISM model card publishes a worked example on the Virchow tile-embedding
file shipped in its own repository, ``tcga/TCGA-B6-A0WZ-...pth``: a [1, 1280]
slide embedding whose first eight values are published, and zero-shot scores
``[[0.0013, 0.9987]]`` for the lobular/ductal prompt pair. The public tools
consume an image tile and therefore cannot read that whole-slide embedding
file; the card example is reproduced directly from the released checkpoint,
and the image tools are checked against the same upstream Virchow and PRISM
computations on the official CONCH H&E tile.
"""

from typing import Any

import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.api.io import decode_as
from caresmcp.api.io.tensor import torch_autocast
from caresmcp.api.models import get_hf_model
from caresmcp.constants import DEVICE
from caresmcp.pathology import prism

NEGATIVE_PROMPTS = ["lobular carcinoma, invasive"]
POSITIVE_PROMPTS = ["ductal carcinoma, invasive"]
TOOLS = [
    pytest.param(prism.prism_slide_embed, {}, "embedding", id="slide-embedding"),
    pytest.param(
        prism.prism_zero_shot_classify,
        {"negative_prompts": NEGATIVE_PROMPTS, "positive_prompts": POSITIVE_PROMPTS},
        "probabilities",
        id="zero-shot",
    ),
]

# Worked example shipped with the model card at this repository revision.
PRISM_REVISION = "b5ef311e78a0811c71ecf618f6f7140e9ebb5fc6"
CARD_EMBEDDING_FILE = (
    "tcga/TCGA-B6-A0WZ-01Z-00-DX1.6CFB236E-36F5-43D6-8DE3-C4ECBD3C14C6.pth"
)
CARD_TILE_SHAPE = (1, 12137, 2560)
CARD_EMBEDDING_SHAPE = (1, 1280)
CARD_LATENT_SHAPE = (1, 512, 1280)
CARD_EMBEDDING_PREFIX = (
    0.0620,
    -0.2983,
    0.5849,
    -0.7383,
    0.1242,
    -0.7440,
    -0.4719,
    -0.2920,
)
CARD_ZERO_SHOT = (0.0013, 0.9987)


@pytest.fixture(scope="module")
def card_sample() -> dict[str, Any]:
    """Load the model card's official Virchow tile-embedding file."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=prism.MODEL_ID,
        filename=CARD_EMBEDDING_FILE,
        revision=PRISM_REVISION,
    )
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def upstream(pathology_image: bytes) -> tuple[Any, Any]:
    """Compute upstream image features independently of the public tools."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    encoder = (
        timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        .to(DEVICE)
        .eval()
    )
    transform = create_transform(
        **resolve_data_config(encoder.pretrained_cfg, model=encoder)
    )
    with torch.inference_mode(), torch_autocast(dtype=torch.float16):
        tokens = encoder(
            transform(decode_as(pathology_image, mode="RGB")).unsqueeze(0).to(DEVICE)
        )
        features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1).float()

    model: Any = get_hf_model(prism.MODEL_ID, trust_remote_code=True)
    with torch.inference_mode(), torch_autocast(dtype=torch.float16):
        representations = model.slide_representations(features.unsqueeze(0).to(DEVICE))
    return model, representations


@pytest.mark.slow
@pytest.mark.cuda
@save_returns("prism", unpack=True)
def test_prism_reproduces_model_card_example(card_sample: dict[str, Any]):
    """Reproduce the card's published embeddings and zero-shot scores."""
    import torch

    tile_embeddings = card_sample["embeddings"].unsqueeze(0).to(DEVICE)
    assert tuple(tile_embeddings.shape) == CARD_TILE_SHAPE

    model: Any = get_hf_model(prism.MODEL_ID, trust_remote_code=True)
    with torch.inference_mode(), torch_autocast(dtype=torch.float16):
        representations = model.slide_representations(tile_embeddings)
        scores = model.zero_shot(
            representations["image_embedding"],
            neg_prompts=NEGATIVE_PROMPTS,
            pos_prompts=POSITIVE_PROMPTS,
        )

    assert tuple(representations["image_embedding"].shape) == CARD_EMBEDDING_SHAPE
    assert tuple(representations["image_latents"].shape) == CARD_LATENT_SHAPE
    prefix = representations["image_embedding"][0, : len(CARD_EMBEDDING_PREFIX)]
    assert prefix.float().cpu().tolist() == pytest.approx(
        CARD_EMBEDDING_PREFIX, abs=2e-3
    )
    assert scores[0].float().cpu().tolist() == pytest.approx(CARD_ZERO_SHOT, abs=2e-3)
    return {
        "image_embedding": representations["image_embedding"].cpu().tolist(),
        "zero_shot": scores.cpu().tolist(),
    }


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize(("tool", "kwargs", "operation"), TOOLS)
@save_returns("prism", unpack=True)
def test_prism_tools(
    tool,
    kwargs: dict[str, Any],
    operation: str,
    pathology_image: bytes,
    upstream: tuple[Any, Any],
):
    """Verify each tool's values against its upstream operation."""
    import torch

    result = tool(image=pathology_image, **kwargs)
    model, representations = upstream
    assert result["model_id"] == prism.MODEL_ID
    assert result["tile_count"] == 1
    with torch.inference_mode(), torch_autocast(dtype=torch.float16):
        if operation == "embedding":
            expected = representations["image_embedding"][0].float().cpu()
            assert result["embedding_size"] == expected.numel()
            torch.testing.assert_close(torch.tensor(result["embedding"]), expected)
        elif operation == "probabilities":
            expected = (
                model.zero_shot(
                    representations["image_embedding"],
                    neg_prompts=NEGATIVE_PROMPTS,
                    pos_prompts=POSITIVE_PROMPTS,
                )[0]
                .float()
                .cpu()
            )
            actual = torch.tensor(
                [result["probability_negative"], result["probability_positive"]]
            )
            torch.testing.assert_close(actual, expected)
            assert actual.sum().item() == pytest.approx(1.0, abs=1e-3)
    return result
