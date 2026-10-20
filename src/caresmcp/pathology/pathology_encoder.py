"""Embedding tools for pathology foundation models."""

from functools import lru_cache
from typing import Any, Literal

from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TASK_EMBED
from caresmcp.tools.image_processing import decode_rgb
from caresmcp.util.moduleloader import requires

PathologyEncoder = Literal[
    "bioptimus/H0-mini",
    "MahmoodLab/UNI",
    "paige-ai/Virchow",
    "paige-ai/Virchow2",
    "bioptimus/H-optimus-0",
]
PathologyPooling = Literal["model", "cls", "mean", "cls_mean"]


@lru_cache(maxsize=2)
def _load_pathology_encoder(model_id: PathologyEncoder) -> tuple[Any, Any]:
    """Load a published pathology encoder and its checkpoint transforms."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torchvision import transforms

    model_kwargs: dict[str, Any] = {"pretrained": True}
    if model_id == "MahmoodLab/UNI":
        model_kwargs.update(init_values=1e-5, dynamic_img_size=True)
    elif model_id == "bioptimus/H-optimus-0":
        model_kwargs.update(init_values=1e-5, dynamic_img_size=False)
    elif model_id in ("paige-ai/Virchow", "paige-ai/Virchow2"):
        from timm.layers import SwiGLUPacked

        model_kwargs.update(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    model = timm.create_model(f"hf-hub:{model_id}", **model_kwargs)
    model = model.to(DEVICE)
    model.eval()
    if model_id == "bioptimus/H-optimus-0":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
    else:
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
    return model, transform


def _extract_pathology_embedding_tensor(
    image: bytes,
    model_id: PathologyEncoder,
    pooling: PathologyPooling | None = None,
) -> tuple[PathologyPooling, Any]:
    """Extract one pathology feature tensor for use by internal tool chains."""
    import torch

    if pooling is None:
        selected_pooling: PathologyPooling = (
            "cls_mean"
            if model_id in ("paige-ai/Virchow", "paige-ai/Virchow2")
            else "model"
        )
    else:
        selected_pooling = pooling
    if selected_pooling not in ("model", "cls", "mean", "cls_mean"):
        raise ValueError("pooling must be 'model', 'cls', 'mean', or 'cls_mean'")
    if (
        model_id not in ("paige-ai/Virchow", "paige-ai/Virchow2")
        and selected_pooling != "model"
    ):
        raise ValueError(
            "cls, mean, and cls_mean pooling are only supported by Virchow"
        )

    try:
        tile = decode_rgb(image)
    except (OSError, ValueError) as exc:
        raise ValueError("image must contain a readable pathology tile") from exc
    if model_id == "bioptimus/H-optimus-0" and tile.size != (224, 224):
        raise ValueError("H-optimus-0 requires 224 by 224 pixel tiles")

    model, transform = _load_pathology_encoder(model_id)
    batch = transform(tile).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model(batch)
    else:
        with torch.inference_mode():
            output = model(batch)

    if (
        not isinstance(output, torch.Tensor)
        or output.ndim not in (2, 3)
        or output.shape[0] != 1
    ):
        raise RuntimeError("pathology encoder returned an invalid feature tensor")

    if output.ndim == 2 and selected_pooling == "model":
        embedding = output
    elif output.ndim == 3 and selected_pooling != "model":
        if output.shape[1] == 0:
            raise RuntimeError("pathology encoder returned no feature tokens")
        if selected_pooling == "cls":
            embedding = output[:, 0, :]
        else:
            patch_token_start = 5 if model_id == "paige-ai/Virchow2" else 1
            if output.shape[1] <= patch_token_start:
                raise RuntimeError("pathology encoder returned no patch tokens")
            patch_tokens = output[:, patch_token_start:, :]
            if selected_pooling == "mean":
                embedding = patch_tokens.mean(dim=1)
            else:
                embedding = torch.cat(
                    (output[:, 0, :], patch_tokens.mean(dim=1)), dim=-1
                )
    else:
        raise RuntimeError(
            "pathology encoder output shape does not match the requested pooling"
        )

    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise RuntimeError("pathology encoder returned an unexpected embedding shape")
    values = embedding[0].detach().float()
    if not torch.isfinite(values).all():
        raise RuntimeError("pathology encoder returned non-finite embedding values")
    return selected_pooling, values


@PATHOLOGY.register("pathology_encoder_embed")
@TASK_EMBED.register("pathology_encoder_embed")
@requires(["torch", "transformers"], extra="hf")
def pathology_encoder_embed(
    image: bytes,
    model_id: PathologyEncoder,
    pooling: PathologyPooling | None = None,
) -> dict[str, Any]:
    """Extract one feature vector from a pathology tile or region of interest.

    The embedding is a model representation, not a diagnosis or calibrated
    disease probability. Check the selected checkpoint's license and access
    requirements before use.

    Args:
        image: Encoded RGB or grayscale pathology tile or region of interest.
        encoder: Hugging Face model ID: ``bioptimus/H0-mini``,
            ``MahmoodLab/UNI``, ``paige-ai/Virchow``, ``paige-ai/Virchow2``,
            or ``bioptimus/H-optimus-0``.
        pooling: For Virchow models, use the class token, mean patch token, or
            their concatenation. If omitted, Virchow uses ``cls_mean`` and the
            other encoders use their model output.

    Returns:
        The model identifier, pooling method, embedding size, and feature vector.

    See Also:
        https://huggingface.co/bioptimus/H0-mini
        https://huggingface.co/MahmoodLab/UNI
        https://huggingface.co/paige-ai/Virchow
        https://huggingface.co/paige-ai/Virchow2
        https://huggingface.co/bioptimus/H-optimus-0
    """
    selected_pooling, values = _extract_pathology_embedding_tensor(
        image=image,
        model_id=model_id,
        pooling=pooling,
    )
    values = values.cpu()

    return {
        "model_id": model_id,
        "pooling": selected_pooling,
        "embedding_size": int(values.numel()),
        "embedding": values.tolist(),
    }
