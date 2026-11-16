"""Compare public PRISM tools with upstream inference on a real pathology image."""

import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import pytest

from caresmcp.api.decorators import save_returns
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


@pytest.fixture(scope="module")
def upstream(pathology_image: bytes) -> tuple[Any, Any]:
    """Compute upstream image features independently of the public tools."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    from caresmcp.api.io import decode_as

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
    with (
        torch.inference_mode(),
        (
            torch.autocast("cuda", dtype=torch.float16)
            if DEVICE == "cuda"
            else nullcontext()
        ),
    ):
        tokens = encoder(
            transform(decode_as(pathology_image, mode="RGB")).unsqueeze(0).to(DEVICE)
        )
        features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1).float()

    model: Any = get_hf_model(prism.MODEL_ID, trust_remote_code=True)
    with (
        torch.inference_mode(),
        (
            torch.autocast("cuda", dtype=torch.float16)
            if DEVICE == "cuda"
            else nullcontext()
        ),
    ):
        representations = model.slide_representations(features.unsqueeze(0).to(DEVICE))
    return model, representations


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize(("tool", "kwargs", "operation"), TOOLS)
@save_returns("prism")
def test_prism_tools(
    tool: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    operation: str,
    pathology_image: bytes,
    upstream: tuple[Any, Any],
):
    """Verify each tool's values against its corresponding upstream operation."""
    import torch

    result = tool(image=pathology_image, **kwargs)
    model, representations = upstream
    json.dumps(result, allow_nan=False)
    assert result["model_id"] == prism.MODEL_ID
    assert result["tile_count"] == 1
    assert result["provenance"]["physical_scale_verified"] is False
    assert result["provenance"]["model_id"] == prism.MODEL_ID
    with (
        torch.inference_mode(),
        (
            torch.autocast("cuda", dtype=torch.float16)
            if DEVICE == "cuda"
            else nullcontext()
        ),
    ):
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
