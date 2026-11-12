from typing import TYPE_CHECKING, Literal

from caresmcp.constants import DEVICE

from .hf_hub import get_hf_checkpoint

if TYPE_CHECKING:
    from torch.nn import Module

MonaiModels = Literal[
    "MONAI/breast_density_classification",
    "MONAI/pathology_tumor_detection",
    "MONAI/pathology_nuclei_classification",
    "MONAI/pathology_nuclei_segmentation_classification",
]

MONAI_MODEL_REVISIONS: dict[MonaiModels, str] = {
    "MONAI/breast_density_classification": "0.1.8",
    "MONAI/pathology_tumor_detection": "0.6.3",
    "MONAI/pathology_nuclei_classification": "0.2.1",
    "MONAI/pathology_nuclei_segmentation_classification": "0.2.7",
}


def _build_monai_model(model_id: MonaiModels) -> "Module":
    """Instantiate the architecture released for a MONAI model-zoo checkpoint."""
    from monai.networks.nets import DenseNet121, HoVerNet, TorchVisionFCModel

    match model_id:
        case "MONAI/breast_density_classification":
            return TorchVisionFCModel(
                model_name="inception_v3",
                num_classes=4,
                pool=None,
                use_conv=False,
                bias=True,
                pretrained=False,
            )
        case "MONAI/pathology_tumor_detection":
            return TorchVisionFCModel(
                model_name="resnet18", num_classes=1, use_conv=True, pretrained=False
            )
        case "MONAI/pathology_nuclei_classification":
            return DenseNet121(spatial_dims=2, in_channels=4, out_channels=4)
        case "MONAI/pathology_nuclei_segmentation_classification":
            return HoVerNet(
                mode="fast", adapt_standard_resnet=True, in_channels=3, out_classes=5
            )
        case _:
            raise ValueError("Unsupported MONAI model")


def get_monai_model(model_id: MonaiModels) -> "Module":
    """Load and cache a released MONAI model-zoo checkpoint for inference."""
    model = _build_monai_model(model_id)
    checkpoint = get_hf_checkpoint(
        model_id,
        "models/model.pt",
        revision=MONAI_MODEL_REVISIONS[model_id],
    )
    # Most bundles wrap the tensors under a "model" entry.
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()
