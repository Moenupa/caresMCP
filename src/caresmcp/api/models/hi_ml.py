"""Matching BioViL image and projected-text encoders."""

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from caresmcp.api.models.hf_hub import get_hf_checkpoint
from caresmcp.constants import LRU_MAX_SIZE

from ..io import device_copy

if TYPE_CHECKING:
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer
    from torch.nn import Module

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


def get_biovil_image_model(model_id: BioViLModel) -> "Module":
    """Load the released image encoder on CPU."""
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
    # get_hf_checkpoint is already lru_cached.
    state = get_hf_checkpoint(model_id, filename, revision=revision)
    model.load_state_dict(state, strict=True)
    return device_copy(model)


def get_biovil_text_model(
    model_id: BioViLModel,
) -> tuple["CXRBertModel", "CXRBertTokenizer"]:
    """Load the matching CXR-BERT tokenizer and joint-space text projector."""
    model, tokenizer = _get_biovil_text_model(model_id)
    return device_copy(model), tokenizer


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_biovil_text_model(
    model_id: BioViLModel,
) -> tuple["CXRBertModel", "CXRBertTokenizer"]:
    """Load and cache the CXR-BERT text encoder and tokenizer on CPU."""
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer

    _, revision = BIOVIL_CHECKPOINTS[model_id]
    tokenizer = CXRBertTokenizer.from_pretrained(model_id, revision=revision)
    model = CXRBertModel.from_pretrained(model_id, revision=revision)
    return model, tokenizer
