"""Image embeddings from the pretrained EchoCare ultrasound foundation model."""

from typing import Annotated, Any

from pydantic import Field

from caresmcp.api.models.hf_hub import _get_hf_checkpoint
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_EMBED, ULTRASOUND
from caresmcp.util.moduleloader import requires

_MODEL_ID = "Moenupa/EchoCare"
_IMAGE_SIZE = 256
ULTRASOUND_IMG = Field(description="Ultrasound image or frame")


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


def _load_echocare_encoder(checkpoint: Any) -> Any:
    """Load and cache the official EchoCare Swin Transformer encoder."""
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
    state_dict = checkpoint.get("state_dict", checkpoint)
    for k in list(dict(state_dict).keys()):
        if k.startswith("swin."):
            state_dict[k[5:]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


@ULTRASOUND.register("echocare_image_embed")
@TASK_EMBED.register("echocare_image_embed")
@requires(["torch", "monai", "transformers"], extra="ultrasound")
def echocare_image_embed(image: Annotated[bytes, ULTRASOUND_IMG]) -> dict[str, Any]:
    """Extract a 2048-dimensional EchoCare embedding from an ultrasound image.

    Returns:
        Model identifier and a 2048-dimensional embedding obtained by global
        average pooling the final Swin Transformer feature map.

    See Also:
        https://github.com/CAIR-HKISI/EchoCare
        https://arxiv.org/abs/2509.11752
    """
    import torch

    tensor = _decode_echocare_image(image, torch)
    checkpoint = _get_hf_checkpoint(model_id=_MODEL_ID, filename="model.safetensors")
    model = _load_echocare_encoder(checkpoint)
    with torch.inference_mode():
        feature_maps = model(tensor)

    final_features = feature_maps[-1]
    if final_features.ndim != 4 or final_features.shape[0] != 1:
        raise RuntimeError("EchoCare encoder returned an invalid final feature map")
    if not torch.isfinite(final_features).all():
        raise RuntimeError("EchoCare encoder returned non-finite features")

    embedding = final_features.mean(dim=(2, 3))[0].detach().float().cpu().tolist()
    if len(embedding) != 2048:
        raise RuntimeError("EchoCare encoder must return a 2048-dimensional embedding")

    return {
        "model": _MODEL_ID,
        "embedding": [float(value) for value in embedding],
    }
