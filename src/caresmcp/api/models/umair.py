"""Load disease-specific models from the Medical Imaging Models release."""

from functools import lru_cache
from typing import Any, Literal, NamedTuple

from caresmcp.constants import DEVICE, LRU_MAX_SIZE

from .hf_hub import get_hf_checkpoint

MODEL_ID = "umairinayat/medical-models"
Disease = Literal[
    "breast_ultrasound",
    "chest_xray",
    "dermatology",
    "endoscopy",
    "fundus_retinography",
    "mammography",
    "microscopy",
    "thyroid_ultrasound",
]
SegmentationDisease = Literal[
    "breast_ultrasound",
    "chest_xray",
    "dermatology",
    "endoscopy",
    "mammography",
    "thyroid_ultrasound",
]


class DiseaseModel(NamedTuple):
    """Model and metadata supplied with its checkpoint."""

    model: Any
    checkpoint: dict[str, Any]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.checkpoint.get("class_names", ()))

    @property
    def mean(self) -> tuple[float, ...]:
        return tuple(self.checkpoint.get("normalization_mean", ()))

    @property
    def std(self) -> tuple[float, ...]:
        return tuple(self.checkpoint.get("normalization_std", ()))


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_umair_classifier(disease: Disease) -> DiseaseModel:
    """Load a disease-specific classifier and its checkpoint metadata."""
    import torch
    from torchvision import models

    checkpoint = get_hf_checkpoint(
        MODEL_ID, f"disease_models/{disease}/classification/best_classifier.pt"
    )
    arch = checkpoint.get("arch", checkpoint.get("architecture"))
    if arch not in ("resnet50", "densenet121", "convnext_small"):
        raise ValueError(f"Unsupported classifier architecture: {arch}")

    model = getattr(models, arch)(weights=None)
    count = len(checkpoint["class_names"])
    if arch == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, count)
    elif arch == "densenet121":
        model.classifier = torch.nn.Linear(model.classifier.in_features, count)
    else:
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, count)

    model.load_state_dict(checkpoint["model_state"], strict=True)
    return DiseaseModel(model.to(DEVICE).eval(), checkpoint)


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_umair_segmentation_model(disease: SegmentationDisease) -> DiseaseModel:
    """Load a disease-specific segmentation model and its checkpoint metadata."""
    checkpoint = get_hf_checkpoint(
        MODEL_ID, f"disease_models/{disease}/segmentation/best_segmenter.pt"
    )
    arch = checkpoint.get("arch", checkpoint.get("architecture"))
    if arch in ("smp_unet", "smp_unetplusplus"):
        import segmentation_models_pytorch as smp

        model_class = smp.Unet if arch == "smp_unet" else smp.UnetPlusPlus
        model = model_class(
            encoder_name=checkpoint.get("encoder", "efficientnet-b3"),
            encoder_weights=None,
            in_channels=checkpoint.get("in_channels", 3),
            classes=checkpoint.get("classes", 1),
        )
    elif arch == "segformer_b3":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig.from_pretrained(
            checkpoint["pretrained"], num_labels=checkpoint["classes"]
        )
        model = SegformerForSemanticSegmentation(config)
    else:
        raise ValueError(f"Unsupported segmentation architecture: {arch}")

    model.load_state_dict(checkpoint["model_state"], strict=True)
    return DiseaseModel(model.to(DEVICE).eval(), checkpoint)
