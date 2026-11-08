"""Academic evaluation helpers for the gated Paige AI PRISM2 model.

PRISM2 is licensed for non-commercial academic research and evaluation only.
The license excludes commercial, clinical, and Research Use Only (RUO) use.
Its outputs must not be used for diagnosis, clinical support, or patient-care
decisions. Access to the checkpoint requires accepting the Hugging Face terms
and is not provided by this package.
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.models import HFModelType, get_hf_model_and_processor
from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TASK_CLS, TASK_EMBED

from .pathology_encoder import (
    _extract_pathology_embedding_tensor,
)

MODEL_ID = "paige-ai/Prism2"
VIRCHOW2_EMBEDDING_SIZE = 1280
_PRISM2_REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "safetensors",
    "einops",
    "timm",
]
Prism2Operation = Literal[
    "response",
    "yes_no_score",
    "base_embedding",
    "diagnostic_embedding",
]
PATHOLOGY_TILE = Field(
    description="Encoded RGB or grayscale pathology tile or ROI sampled at 20x magnification."
)
PRISM2_EMBEDDING_TYPE = Field(description="PRISM2 embedding type to return.")
YES_NO_QUESTION = Field(min_length=1, description="A single yes/no question to score.")
CAPTION_PROMPT = Field(
    min_length=1, description="A single-turn prompt for pathology image generation."
)
MAX_NEW_TOKENS = Field(
    ge=1, le=4096, description="Maximum number of new tokens to generate."
)


def _prepare_prism2_batch(image: bytes) -> tuple[Any, Any]:
    """Encode one pathology tile with Virchow2 and prepare a PRISM2 batch."""
    import torch

    if DEVICE != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PRISM2 requires a CUDA GPU with bfloat16 support")

    _, embedding = _extract_pathology_embedding_tensor(
        image=image,
        model_id="paige-ai/Virchow2",
        pooling="cls",
    )
    if embedding.shape != (VIRCHOW2_EMBEDDING_SIZE,):
        raise RuntimeError("Virchow2 returned an embedding with an unexpected shape")

    model, processor = get_hf_model_and_processor(
        MODEL_ID,
        model_type=HFModelType.AUTO,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    slide_embeddings = embedding.unsqueeze(0).cpu().contiguous()
    batch = processor(tile_embeddings=[slide_embeddings]).to(DEVICE)
    return model, batch


def _run_prism2_operation(
    image: bytes,
    operation: Prism2Operation = "response",
    prompt: str | None = None,
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """Run a requested PRISM2 operation on one pathology tile or ROI."""
    import torch

    if operation in ("response", "yes_no_score") and (
        prompt is None or not prompt.strip()
    ):
        raise ValueError("prompt must not be empty for this operation")
    if operation == "response" and not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096")

    model, batch = _prepare_prism2_batch(image)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if operation == "base_embedding":
            embedding = model.get_base_embedding(**batch)
        elif operation == "diagnostic_embedding":
            embedding = model.get_diagnostic_embedding(**batch)
        elif operation == "yes_no_score":
            scores = model.yes_no_score(
                tile_embeddings=batch["tile_embeddings"],
                attention_mask=batch["attention_mask"],
                question=prompt or "",
            )
        else:
            responses = model.get_response(
                **batch,
                prompt=prompt or "",
                max_new_tokens=max_new_tokens,
            )

    if operation in ("base_embedding", "diagnostic_embedding"):
        expected_size = 2560 if operation == "base_embedding" else 3072
        if embedding.ndim != 2 or embedding.shape != (1, expected_size):
            raise RuntimeError("PRISM2 returned an embedding with an unexpected shape")
        return {
            "model_id": MODEL_ID,
            "operation": operation,
            "embedding": embedding[0].detach().float().cpu().tolist(),
        }
    if operation == "yes_no_score":
        if scores.numel() != 1:
            raise RuntimeError(
                "PRISM2 returned an unexpected number of question scores"
            )
        return {
            "model_id": MODEL_ID,
            "operation": operation,
            "prompt": prompt,
            "probability_yes": float(scores.reshape(-1)[0].detach().float().cpu()),
        }

    if not isinstance(responses, (list, tuple)) or len(responses) != 1:
        raise RuntimeError("PRISM2 returned an unexpected response batch")
    return {
        "model_id": MODEL_ID,
        "operation": operation,
        "response": str(responses[0]),
    }


@PATHOLOGY.register("prism2_image_embed")
@TASK_EMBED.register("prism2_image_embed")
@requires(_PRISM2_REQUIRED_PACKAGES, extra="pathology")
def prism2_image_embed(
    image: Annotated[bytes, PATHOLOGY_TILE],
    embedding_type: Annotated[
        Literal["base", "diagnostic"], PRISM2_EMBEDDING_TYPE
    ] = "base",
) -> dict[str, Any]:
    """Encode a pathology tile or ROI as a PRISM2 embedding.

    See Also:
        https://huggingface.co/paige-ai/Prism2
        https://huggingface.co/paige-ai/Virchow2
        https://doi.org/10.1038/s41591-026-04521-4
    """
    if embedding_type not in ("base", "diagnostic"):
        raise ValueError("embedding_type must be 'base' or 'diagnostic'")

    operation: Prism2Operation = (
        "base_embedding" if embedding_type == "base" else "diagnostic_embedding"
    )
    return _run_prism2_operation(image, operation=operation)


@PATHOLOGY.register("prism2_score_yes_no_question")
@TASK_CLS.register("prism2_score_yes_no_question")
@requires(_PRISM2_REQUIRED_PACKAGES, extra="pathology")
def prism2_score_yes_no_question(
    image: Annotated[bytes, PATHOLOGY_TILE],
    question: Annotated[str, YES_NO_QUESTION],
) -> dict[str, Any]:
    """Score how likely a yes/no answer is for a pathology tile or ROI.

    See Also:
        https://huggingface.co/paige-ai/Prism2
    """
    return _run_prism2_operation(
        image,
        operation="yes_no_score",
        prompt=question,
    )


@PATHOLOGY.register("prism2_image_caption")
@requires(_PRISM2_REQUIRED_PACKAGES, extra="hf")
def prism2_image_caption(
    image: Annotated[bytes, PATHOLOGY_TILE],
    prompt: Annotated[str, CAPTION_PROMPT],
    max_new_tokens: Annotated[int, MAX_NEW_TOKENS] = 200,
) -> dict[str, Any]:
    """Generate text about a pathology tile or ROI from a prompt.

    See Also:
        https://huggingface.co/paige-ai/Prism2
    """
    return _run_prism2_operation(
        image,
        operation="response",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
