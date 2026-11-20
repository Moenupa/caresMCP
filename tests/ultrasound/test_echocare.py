"""Check EchoCare embeddings against the authors' released checkpoint.

EchoCare's published scores are downstream task results. Its LVEF regression
claim (MAE 3.91 on CAMUS, arXiv:2509.11752) needs a fine-tuned regression head
and the CAMUS benchmark, neither of which is part of the released encoder, and
the repository releases no sample image or reference embedding. This test
instead loads the official ``CAIR-HKISI/EchoCare`` checkpoint exactly as its
model card shows and compares the resulting final-feature mean with the tool's
embedding on a frame from EchoNet-Dynamic's official release.
"""

from typing import Any

import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.ultrasound.echocare import echocare_image_embed

# Revision-pinned official checkpoint. The Hub file is a torch serialization
# artifact named ``.safetensors``; the tool reads the safetensors mirror
# ``Moenupa/EchoCare``.
ECHOCARE_MODEL_ID = "CAIR-HKISI/EchoCare"
ECHOCARE_REVISION = "010b9b486a06e243f6b63b9d5cd57c64731f4419"
ECHOCARE_FILENAME = "model.safetensors"
# Architecture from the repository's ``config.json`` and README usage snippet.
ECHOCARE_CONFIG: dict[str, Any] = {
    "in_chans": 3,
    "embed_dim": 128,
    "window_size": [8, 8],
    "patch_size": [2, 2],
    "depths": [2, 2, 18, 2],
    "num_heads": [4, 8, 16, 32],
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "drop_path_rate": 0.0,
    "use_checkpoint": False,
    "spatial_dims": 2,
    "use_v2": True,
}
IMAGE_SIZE = 256


@pytest.mark.slow
@pytest.mark.cuda
@skip_on_exception(ImportError, OSError)
@save_returns()
def test_embedding_matches_released_checkpoint(echonet_png):
    """Reproduce the released encoder's final-feature mean."""
    import torch
    from huggingface_hub import hf_hub_download
    from monai.networks.nets.swin_unetr import SwinTransformer
    from torchvision import transforms

    from caresmcp.api.io import decode_as
    from caresmcp.constants import DEVICE

    embedding = echocare_image_embed(echonet_png)["embedding"]

    # The released checkpoint holds tensors only, so weights_only is sufficient.
    # It is a torch archive despite its ``.safetensors`` name, so load through
    # a file object to bypass torch's suffix-based safetensors dispatch.
    with open(
        hf_hub_download(
            repo_id=ECHOCARE_MODEL_ID,
            filename=ECHOCARE_FILENAME,
            revision=ECHOCARE_REVISION,
        ),
        "rb",
    ) as checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    # Normalize the released encoder state the way the release README and the
    # tool do: unwrap ``state_dict``, drop the pretraining ``mask_token``, and
    # strip the ``swin.`` wrapper prefix.
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict.pop("mask_token", None)
    state_dict = {key.removeprefix("swin."): value for key, value in state_dict.items()}
    encoder = SwinTransformer(**ECHOCARE_CONFIG)
    encoder.load_state_dict(state_dict)
    encoder.to(DEVICE).eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )
    with torch.inference_mode():
        feature_maps = encoder(
            transform(decode_as(echonet_png, mode="RGB")).unsqueeze(0).to(DEVICE)
        )
    expected = feature_maps[-1].mean(dim=(2, 3))[0].float().cpu().tolist()

    assert embedding == pytest.approx(expected, rel=1e-5, abs=1e-5)
    return {"embedding": embedding}
