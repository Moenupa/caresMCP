import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from caresmcp.constants import DEVICE, LRU_MAX_SIZE

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin


class HFModelType(Enum):
    """Transformers model and processor class combinations used internally."""

    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_ENCODER = "image_encoder"
    AUTO = "auto"
    SAM = "sam"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_hf_model_and_processor(
    model_id: str,
    model_type: HFModelType = HFModelType.IMAGE_CLASSIFICATION,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
    *,
    subfolder: str = "",
    processor_subfolder: str | None = None,
) -> "tuple[PreTrainedModel, ProcessorMixin]":
    """Load and cache a Transformers model and its processor.

    Image-classification checkpoints use their task-specific model and image
    processor classes by default. Image encoders use ``AutoModel`` with
    ``AutoImageProcessor``. ``HFModelType.AUTO`` uses ``AutoModel`` and
    ``AutoProcessor`` for checkpoints with custom model interfaces.

    ``subfolder`` selects a model head. Its processor is loaded from the same
    folder unless ``processor_subfolder`` explicitly selects another folder.

    Never support revision or local_files_only. Since it brings complexity we/agents dont need.
    """
    match model_type:
        case HFModelType.SAM:
            from transformers import SamModel, SamProcessor

            model_class = SamModel
            processor_class = SamProcessor
        case HFModelType.SEMANTIC_SEGMENTATION:
            from transformers import (
                AutoImageProcessor,
                AutoModelForSemanticSegmentation,
            )

            model_class = AutoModelForSemanticSegmentation
            processor_class = AutoImageProcessor
        case HFModelType.IMAGE_CLASSIFICATION:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            model_class = AutoModelForImageClassification
            processor_class = AutoImageProcessor
        case HFModelType.IMAGE_ENCODER:
            from transformers import AutoImageProcessor, AutoModel

            model_class = AutoModel
            processor_class = AutoImageProcessor
        case HFModelType.AUTO:
            from transformers import AutoModel, AutoProcessor

            model_class = AutoModel
            processor_class = AutoProcessor
        case _:
            raise ValueError("model_type must be an HFModelType member")

    model_kwargs: dict[str, Any] = {}
    processor_kwargs: dict[str, Any] = {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
        processor_kwargs["trust_remote_code"] = True
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if subfolder:
        model_kwargs["subfolder"] = subfolder
    processor_folder = subfolder if processor_subfolder is None else processor_subfolder
    if processor_folder:
        processor_kwargs["subfolder"] = processor_folder

    processor = processor_class.from_pretrained(model_id, **processor_kwargs)
    model = model_class.from_pretrained(model_id, **model_kwargs).to(DEVICE)
    model.eval()
    return model, processor


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_hf_checkpoint(
    model_id: str,
    filename: str,
    *,
    revision: str | None = None,
    map_location: str = "cpu",
) -> Any:
    """Load a tensor-only PyTorch checkpoint from a Hub repository."""
    import torch
    from huggingface_hub import hf_hub_download

    artifact = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        revision=revision,
    )
    # safetensors handled in torch.load()
    return torch.load(artifact, map_location=map_location, weights_only=True)


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_hf_json(model_id: str, filename: str) -> Any:
    """Load and cache JSON metadata shipped alongside Hub model weights."""
    from huggingface_hub import hf_hub_download

    artifact = hf_hub_download(repo_id=model_id, filename=filename)
    return json.loads(Path(artifact).read_text())
