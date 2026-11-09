"""Embedding tools for pathology foundation models."""

from typing import Annotated, Any, Literal

from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.imageio import decode_as
from caresmcp.api.models import (
    HFModelType,
    get_hf_model_and_processor,
    get_timm_model,
)
from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TASK_EMB

PathologyEncoder = Literal[
    "bioptimus/H0-mini",
    "MahmoodLab/UNI",
    "MahmoodLab/UNI2-h",
    "owkin/phikon",
    "owkin/phikon-v2",
    "histai/hibou-b",
    "paige-ai/Virchow",
    "paige-ai/Virchow2",
    "bioptimus/H-optimus-0",
]
PathologyPooling = Literal["model", "cls", "mean", "cls_mean"]
PATHOLOGY_TILE = Field(description="Encoded RGB or grayscale pathology tile or ROI.")
PATHOLOGY_ENCODER = Field(description="Supported Hugging Face pathology encoder.")
PATHOLOGY_POOLING = Field(description="Token pooling method for the selected encoder.")


def _load_pathology_encoder(model_id: PathologyEncoder) -> tuple[Any, Any]:
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torchvision import transforms

    if model_id in ("owkin/phikon", "owkin/phikon-v2", "histai/hibou-b"):
        model, processor = get_hf_model_and_processor(
            model_id,
            model_type=HFModelType.IMAGE_ENCODER,
            trust_remote_code=model_id == "histai/hibou-b",
            use_fast=False,
        )

        def transform(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]

        return model, transform

    model_kwargs: dict[str, Any] = {}
    if model_id == "MahmoodLab/UNI":
        model_kwargs.update(init_values=1e-5, dynamic_img_size=True)
    elif model_id == "MahmoodLab/UNI2-h":
        from timm.layers import SwiGLUPacked

        model_kwargs.update(
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
    elif model_id == "bioptimus/H-optimus-0":
        model_kwargs.update(init_values=1e-5, dynamic_img_size=False)
    elif model_id in ("paige-ai/Virchow", "paige-ai/Virchow2"):
        from timm.layers import SwiGLUPacked

        model_kwargs.update(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    model = get_timm_model(f"hf-hub:{model_id}", **model_kwargs)
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
    image: bytes | Image.Image,
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

    tile = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else decode_as(image, mode="RGB")
    )
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

    if model_id in ("owkin/phikon", "owkin/phikon-v2", "histai/hibou-b"):
        output = output.last_hidden_state[:, 0, :]

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


@PATHOLOGY.register("pathology_image_embed")
@TASK_EMB.register("pathology_image_embed")
@requires(["torch", "torchvision", "timm", "transformers"], extra="pathology")
def pathology_image_embed(
    image: Annotated[bytes, PATHOLOGY_TILE],
    model_id: Annotated[PathologyEncoder, PATHOLOGY_ENCODER],
    pooling: Annotated[PathologyPooling | None, PATHOLOGY_POOLING] = None,
) -> dict[str, Any]:
    """Encode a pathology tile or ROI as a feature vector.

    See Also:
        https://huggingface.co/bioptimus/H0-mini
        https://huggingface.co/MahmoodLab/UNI
        https://huggingface.co/MahmoodLab/UNI2-h
        https://huggingface.co/owkin/phikon
        https://huggingface.co/owkin/phikon-v2
        https://huggingface.co/histai/hibou-b
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
