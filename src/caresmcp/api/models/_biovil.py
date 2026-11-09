"""Cached matching BioViL image and projected-text encoders."""

from typing import Any, Literal

from caresmcp.api.models.hf_hub import get_hf_checkpoint
from caresmcp.constants import DEVICE

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


def get_biovil_image_model(model_id: BioViLModel) -> Any:
    """Load the released image encoder and joint-space projector."""
    from health_multimodal.image.model.model import ImageModel, MultiImageModel
    from health_multimodal.image.model.types import ImageEncoderType

    temporal = model_id == "microsoft/BiomedVLP-BioViL-T"
    factory = MultiImageModel if temporal else ImageModel
    model = factory(
        img_encoder_type=(
            ImageEncoderType.RESNET50_MULTI_IMAGE
            if temporal
            else ImageEncoderType.RESNET50
        ),
        joint_feature_size=128,
    )
    filename, revision = BIOVIL_CHECKPOINTS[model_id]
    state = get_hf_checkpoint(model_id, filename, revision=revision)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def get_biovil_text_model(model_id: BioViLModel) -> tuple[Any, Any]:
    """Load the matching CXR-BERT tokenizer and joint-space text projector."""
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer

    _, revision = BIOVIL_CHECKPOINTS[model_id]
    tokenizer = CXRBertTokenizer.from_pretrained(model_id, revision=revision)
    model = CXRBertModel.from_pretrained(model_id, revision=revision)
    return model.to(DEVICE).eval(), tokenizer
