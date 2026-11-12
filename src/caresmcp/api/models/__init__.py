from .hf_hub import (
    HFModelType,
    get_hf_checkpoint,
    get_hf_model,
    get_hf_model_and_processor,
)
from .monai import get_monai_model
from .timm import get_timm_model
from .torch_hub import get_torch_model
from .torchxrayvision import get_torchxrayvision_model

HFMODEL_ID_DESC = "Huggingface Model ID"
SOFTMAX_DESC = "Sigmoid for multi-label; softmax for exclusive classes."

__all__ = [
    "HFMODEL_ID_DESC",
    "SOFTMAX_DESC",
    "HFModelType",
    "get_hf_checkpoint",
    "get_hf_model",
    "get_hf_model_and_processor",
    "get_monai_model",
    "get_timm_model",
    "get_torch_model",
    "get_torchxrayvision_model",
]
