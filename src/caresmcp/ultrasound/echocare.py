"""Image embeddings from the pretrained EchoCare ultrasound foundation model."""

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as
from caresmcp.api.models import get_hf_checkpoint
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_EMB, ULTRASOUND

if TYPE_CHECKING:
    from torchvision.transforms import Compose

_MODEL_ID = "Moenupa/EchoCare"
_IMAGE_SIZE = 256
ULTRASOUND_IMG = Field(description="Ultrasound image or frame")


def _echocare_transform() -> "Compose":
    """Return the EchoCare input transform for RGB ultrasound frames."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (_IMAGE_SIZE, _IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )


def _load_echocare_encoder() -> Any:
    """Load and cache the pretrained EchoCare encoder."""
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
    checkpoint = get_hf_checkpoint(model_id=_MODEL_ID, filename="model.safetensors")
    checkpoint_state = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key.removeprefix("swin."): value for key, value in checkpoint_state.items()
    }

    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


@ULTRASOUND.register("echocare_image_embed")
@TASK_EMB.register("echocare_image_embed")
@requires(["torch", "torchvision", "monai", "transformers"], extra="ultrasound")
def echocare_image_embed(image: Annotated[bytes, ULTRASOUND_IMG]) -> dict[str, Any]:
    """Extract a 2048-dimensional EchoCare embedding from an ultrasound image.

    See Also:
        https://github.com/CAIR-HKISI/EchoCare
        https://arxiv.org/abs/2509.11752
    """
    import torch

    model = _load_echocare_encoder()
    with torch.inference_mode():
        feature_maps = model(
            _echocare_transform()(decode_as(image, mode="RGB")).unsqueeze(0).to(DEVICE)
        )

    final_features = feature_maps[-1]

    embedding = final_features.mean(dim=(2, 3))[0].detach().float().cpu().tolist()
    return {
        "model": "CAIR-HKISI/EchoCare",
        "embedding": [float(value) for value in embedding],
    }
