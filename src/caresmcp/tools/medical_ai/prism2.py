"""Academic evaluation helpers for the gated Paige AI PRISM2 model.

PRISM2 is licensed for non-commercial academic research and evaluation only.
The license excludes commercial, clinical, and Research Use Only (RUO) use.
Its outputs must not be used for diagnosis, clinical support, or patient-care
decisions. Access to the checkpoint requires accepting the Hugging Face terms
and is not provided by this package.
"""

from io import BytesIO
from typing import Any, Literal

from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TAG_PATHOLOGY
from caresmcp.tools.medical_ai.transformers import get_hf_model_and_processor
from caresmcp.util.moduleloader import requires

MODEL_ID = "paige-ai/Prism2"
VIRCHOW2_EMBEDDING_SIZE = 1280
_PRISM2_REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "safetensors",
    "einops",
    "flash_attn",
]
Prism2Operation = Literal[
    "response",
    "yes_no_score",
    "base_embedding",
    "diagnostic_embedding",
]


def _prepare_prism2_batch(tile_embeddings: bytes) -> tuple[Any, Any]:
    """Decode one NumPy tile-embedding array and prepare a PRISM2 batch."""
    import numpy as np
    import torch

    try:
        embeddings = np.load(BytesIO(tile_embeddings), allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError("tile_embeddings must contain a NumPy .npy array") from exc

    if not isinstance(embeddings, np.ndarray):
        raise TypeError("tile_embeddings must contain one .npy array")
    if embeddings.ndim != 2 or embeddings.shape[1] != VIRCHOW2_EMBEDDING_SIZE:
        raise ValueError(
            "tile_embeddings must have shape (N, 1280) for Virchow2 class-token features"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("tile_embeddings must contain at least one tile")
    if embeddings.dtype.kind != "f":
        raise ValueError("tile_embeddings must contain floating-point values")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("tile_embeddings must contain only finite values")

    if DEVICE != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PRISM2 requires a CUDA GPU with bfloat16 support")

    model, processor = get_hf_model_and_processor(
        MODEL_ID,
        model_type="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    slide_embeddings = torch.from_numpy(embeddings)
    batch = processor(tile_embeddings=[slide_embeddings]).to(DEVICE)
    return model, batch


def _run_prism2_operation(
    tile_embeddings: bytes,
    operation: Prism2Operation = "response",
    prompt: str | None = None,
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """Run a requested PRISM2 operation on one slide's tile embeddings."""
    import torch

    if operation not in (
        "response",
        "yes_no_score",
        "base_embedding",
        "diagnostic_embedding",
    ):
        raise ValueError(
            "operation must be 'response', 'yes_no_score', 'base_embedding', "
            "or 'diagnostic_embedding'"
        )
    if operation in ("response", "yes_no_score") and (
        prompt is None or not prompt.strip()
    ):
        raise ValueError("prompt must not be empty for this operation")
    if operation == "response" and not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096")

    model, batch = _prepare_prism2_batch(tile_embeddings)
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


@MEDAI.register("prism2_extract_embedding")
@TAG_PATHOLOGY.register("prism2_extract_embedding")
@requires(_PRISM2_REQUIRED_PACKAGES, extra="prism2")
def prism2_extract_embedding(
    tile_embeddings: bytes,
    embedding_type: Literal["base", "diagnostic"] = "base",
) -> dict[str, Any]:
    """Extract a PRISM2 slide embedding from Virchow2 tile features.

    This tool is for non-commercial academic research and evaluation only. The
    license excludes commercial, clinical, and Research Use Only (RUO) use.
    Embeddings must not support diagnosis or patient-care decisions.

    Args:
        tile_embeddings: NumPy ``.npy`` bytes containing an ``(N, 1280)`` array
            of Virchow2 class-token features using 20x magnification, 224x224
            tiles, and foreground tissue tiles.
        embedding_type: ``"base"`` returns a 2560-dimensional slide embedding;
            ``"diagnostic"`` returns a 3072-dimensional language-model embedding.

    Returns:
        The model identifier, embedding type, and embedding values.

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
    return _run_prism2_operation(tile_embeddings, operation=operation)


@MEDAI.register("prism2_score_yes_no_question")
@TAG_PATHOLOGY.register("prism2_score_yes_no_question")
@requires(_PRISM2_REQUIRED_PACKAGES, extra="prism2")
def prism2_score_yes_no_question(
    tile_embeddings: bytes,
    question: str,
) -> dict[str, Any]:
    """Score a yes/no question for one slide with PRISM2.

    This tool is for non-commercial academic research and model evaluation only.
    The license excludes commercial, clinical, and Research Use Only (RUO) use.
    The score must not be used for diagnosis, clinical support, or patient-care
    decisions.

    Args:
        tile_embeddings: NumPy ``.npy`` bytes containing an ``(N, 1280)`` array
            of Virchow2 class-token features using 20x magnification, 224x224
            tiles, and foreground tissue tiles.
        question: A single yes/no question to score.

    Returns:
        The model identifier, question, and conditional probability of Yes.

    See Also:
        https://huggingface.co/paige-ai/Prism2
    """
    return _run_prism2_operation(
        tile_embeddings,
        operation="yes_no_score",
        prompt=question,
    )


@MEDAI.register("prism2_generate_response")
@TAG_PATHOLOGY.register("prism2_generate_response")
@requires(_PRISM2_REQUIRED_PACKAGES, extra="prism2")
def prism2_generate_response(
    tile_embeddings: bytes,
    prompt: str,
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """Generate a single-turn PRISM2 response for academic model evaluation.

    Generated text may contain errors. This tool is for non-commercial academic
    research and evaluation only; the license excludes commercial, clinical,
    and Research Use Only (RUO) use. Responses must not be used for diagnosis,
    clinical support, or patient-care decisions.

    Args:
        tile_embeddings: NumPy ``.npy`` bytes containing an ``(N, 1280)`` array
            of Virchow2 class-token features using 20x magnification, 224x224
            tiles, and foreground tissue tiles.
        prompt: A single-turn prompt. Use ``"Write a report"`` for report
            generation or provide a single question for question answering.
        max_new_tokens: Maximum number of new tokens to generate, from 1 to 4096.

    Returns:
        The model identifier and generated response text.

    See Also:
        https://huggingface.co/paige-ai/Prism2
    """
    return _run_prism2_operation(
        tile_embeddings,
        operation="response",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
