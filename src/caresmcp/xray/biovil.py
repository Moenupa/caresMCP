"""BioViL static and temporal chest radiograph embeddings."""

from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.imageio import decode_as
from caresmcp.api.models import get_hf_checkpoint
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_EMBED, XRAY

BioViLModel = Literal[
    "microsoft/BiomedVLP-CXR-BERT-specialized",
    "microsoft/BiomedVLP-BioViL-T",
]

BIOVIL_CHECKPOINTS = {
    "microsoft/BiomedVLP-CXR-BERT-specialized": (
        "biovil_image_resnet50_proj_size_128.pt",
        "v1.1",
    ),
    "microsoft/BiomedVLP-BioViL-T": ("biovil_t_image_model_proj_size_128.pt", "v1.0"),
}


def _biovil_image_tensor(image: bytes, crop_size: int):
    """Apply the published min/max intensity remapping and image transforms."""
    from health_multimodal.image.data.io import remap_to_uint8
    from health_multimodal.image.data.transforms import (
        create_chest_xray_transform_for_inference,
    )

    source = decode_as(image)
    pixels = np.asarray(source.convert("RGB") if source.mode == "P" else source)
    grayscale = Image.fromarray(remap_to_uint8(pixels)).convert("L")
    transform = create_chest_xray_transform_for_inference(512, crop_size)
    return transform(grayscale).unsqueeze(0).to(DEVICE)


@XRAY.register("biovil_image_embed")
@TASK_EMBED.register("biovil_image_embed")
@requires(["torch", "health_multimodal", "transformers"], extra="xray")
def biovil_image_embed(
    image: Annotated[bytes, Field(description="Current chest radiograph image")],
    model_id: Annotated[
        BioViLModel,
        Field(description="Official BioViL or BioViL-T image checkpoint repository"),
    ] = "microsoft/BiomedVLP-BioViL-T",
    prior_image: Annotated[
        bytes | None,
        Field(
            description="Earlier chest radiograph from the same patient; BioViL-T only."
        ),
    ] = None,
) -> dict:
    """Embed a chest radiograph with BioViL, or a current/prior pair with BioViL-T.

    The normalized 128-dimensional vector is in the model's projected joint
    image-text space. It has no diagnosis head. Compare vectors only from the
    same model and temporal input configuration. BioViL-T uses its learned
    missing-prior representation when no prior image is supplied.

    See Also:
        https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal
        https://huggingface.co/microsoft/BiomedVLP-BioViL-T
    """
    import torch
    from health_multimodal.image.model.model import ImageModel, MultiImageModel
    from health_multimodal.image.model.types import ImageEncoderType

    temporal = model_id == "microsoft/BiomedVLP-BioViL-T"
    if prior_image is not None and not temporal:
        raise ValueError("prior_image is supported only by BioViL-T")
    crop_size = 448 if temporal else 480
    current = _biovil_image_tensor(image, crop_size)
    filename, revision = BIOVIL_CHECKPOINTS[model_id]
    factory = MultiImageModel if temporal else ImageModel
    model = factory(
        img_encoder_type=(
            ImageEncoderType.RESNET50_MULTI_IMAGE
            if temporal
            else ImageEncoderType.RESNET50
        ),
        joint_feature_size=128,
    )
    state = get_hf_checkpoint(model_id, filename, revision=revision)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()
    with torch.inference_mode():
        if temporal:
            previous = (
                _biovil_image_tensor(prior_image, crop_size)
                if prior_image is not None
                else None
            )
            output = model(current_image=current, previous_image=previous)
        else:
            output = model(current)
        vector = output.projected_global_embedding[0].float()
        vector = torch.nn.functional.normalize(vector, dim=0)
    return {
        "model_id": model_id,
        "checkpoint": filename,
        "checkpoint_revision": revision,
        "prior_image_used": prior_image is not None,
        "representation": "l2_normalized_projected_global_embedding",
        "embedding": vector.cpu().tolist(),
    }
