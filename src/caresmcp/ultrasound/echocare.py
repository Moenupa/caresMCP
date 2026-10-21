"""Image embeddings from the pretrained EchoCare ultrasound foundation model."""

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_EMBED, ULTRASOUND
from caresmcp.util.moduleloader import requires

EchoCareModelId = Literal["CAIR-HKISI/EchoCare"]
_MODEL_ID: EchoCareModelId = "CAIR-HKISI/EchoCare"
_IMAGE_SIZE = 256
_STAGE_NAMES = ("patch_embed", "stage_1", "stage_2", "stage_3", "stage_4")


def _decode_echocare_image(image: bytes, torch: Any) -> Any:
    """Decode an RGB image and apply the EchoCare input transform."""
    from io import BytesIO

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    if not image:
        raise ValueError("image must contain a readable ultrasound frame")

    try:
        with Image.open(BytesIO(image)) as source:
            resized = source.convert("RGB").resize(
                (_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("image must contain a readable ultrasound frame") from exc

    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


@lru_cache(maxsize=2)
def _load_echocare_encoder(checkpoint_path: str) -> Any:
    """Load and cache the official EchoCare Swin Transformer encoder."""
    import torch
    from monai.networks.nets.swin_unetr import SwinTransformer

    model = SwinTransformer(
        in_chans=3,
        embed_dim=128,
        window_size=[8, 8],
        patch_size=[2, 2],
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=2,
        use_v2=True,
    )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint must contain a state dictionary")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint must contain a state dictionary")

        state_dict = dict(state_dict)
        if any(name.startswith("module.") for name in state_dict):
            state_dict = {
                name.removeprefix("module."): value
                for name, value in state_dict.items()
            }
        state_dict.pop("mask_token", None)
        if any(name.startswith("encoder.") for name in state_dict):
            state_dict = {
                name.removeprefix("encoder."): value
                for name, value in state_dict.items()
                if name.startswith("encoder.")
            }
        model.load_state_dict(state_dict, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "checkpoint_path must point to the official EchoCare encoder weights"
        ) from exc

    return model.to(DEVICE).eval()


@ULTRASOUND.register("echocare_image_embed")
@TASK_EMBED.register("echocare_image_embed")
@requires(["torch", "monai", "PIL"], extra="ultrasound")
def echocare_image_embed(
    image: bytes,
    checkpoint_path: str,
    model_id: EchoCareModelId = _MODEL_ID,
) -> dict[str, Any]:
    """Extract a 2048-dimensional EchoCare embedding from an ultrasound image.

    The official EchoCare checkpoint is distributed from the project page.
    Download its pretrained encoder weights and provide the local checkpoint
    path. The image is resized to 256 by 256 RGB pixels and scaled to the
    0-to-1 range.

    Args:
        image: Encoded ultrasound image or frame bytes.
        checkpoint_path: Local path to the downloaded EchoCare encoder weights.
        model_id: Supported EchoCare model identifier.

    Returns:
        Model identifier and a 2048-dimensional embedding obtained by global
        average pooling the final Swin Transformer feature map.

    See Also:
        https://github.com/CAIR-HKISI/EchoCare
        https://arxiv.org/abs/2509.11752
    """
    import torch

    if model_id != _MODEL_ID:
        raise ValueError("model_id must be 'CAIR-HKISI/EchoCare'")

    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint_path.strip() or not checkpoint.is_file():
        raise ValueError("checkpoint_path must be a readable local file")

    tensor = _decode_echocare_image(image, torch)
    model = _load_echocare_encoder(str(checkpoint.resolve()))
    with torch.inference_mode():
        feature_maps = model(tensor)

    if not isinstance(feature_maps, list) or len(feature_maps) != len(_STAGE_NAMES):
        raise RuntimeError("EchoCare encoder must return five feature maps")

    final_features = feature_maps[-1]
    if final_features.ndim != 4 or final_features.shape[0] != 1:
        raise RuntimeError("EchoCare encoder returned an invalid final feature map")
    if not torch.isfinite(final_features).all():
        raise RuntimeError("EchoCare encoder returned non-finite features")

    embedding = final_features.mean(dim=(2, 3))[0].detach().float().cpu().tolist()
    if len(embedding) != 2048:
        raise RuntimeError("EchoCare encoder must return a 2048-dimensional embedding")

    return {
        "model": model_id,
        "embedding": [float(value) for value in embedding],
    }
