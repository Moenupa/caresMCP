"""Load Transformers models, processors, and Hub checkpoint files."""

import json
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from caresmcp.constants import LRU_MAX_SIZE

from ..io import device_copy

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin


class HFModelType(Enum):
    """Model and processor class pairings for Hub checkpoints.

    Each member pairs one model class with one processor class for
    :func:`get_hf_model_and_processor`, fixing the loaded model's interface
    and therefore the fields its forward pass returns.
    """

    IMAGE_CLASSIFICATION = auto()
    IMAGE_ENCODER = auto()
    IMAGE_PRETRAINING = auto()
    AUTO = auto()
    SAM = auto()
    SAM2 = auto()
    SEMANTIC_SEGMENTATION = auto()


# Model loading intentionally omits `revision` and `local_files_only`; both add
# complexity that the current tooling does not need.
def get_hf_model(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
    subfolder: str = "",
) -> "PreTrainedModel":
    """Load a Hub checkpoint through :class:`~transformers.AutoModel`.

    Use this for checkpoints that need no bundled processor, such as models
    with a repository-defined interface. Loaded weights are cached on the host;
    each call receives an independent evaluation-mode copy on
    ``caresmcp.constants.DEVICE``.

    Args:
        model_id: Hub repository ID of the checkpoint.
        trust_remote_code: Load the model implementation shipped in the
            repository instead of the one in the installed ``transformers``.
        torch_dtype: Weight dtype passed to ``from_pretrained``, for example
            ``"auto"`` or ``"bfloat16"``. ``None`` keeps the checkpoint dtype.
        subfolder: Repository subfolder that contains the weights.

    Returns:
        The model on ``caresmcp.constants.DEVICE`` in evaluation mode.
    """
    model = _get_hf_model(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        subfolder=subfolder,
    )
    return device_copy(model)


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_hf_model(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
    subfolder: str = "",
) -> "PreTrainedModel":
    """Load and cache a processor-free checkpoint through AutoModel on CPU."""
    from transformers import AutoModel

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if subfolder:
        kwargs["subfolder"] = subfolder
    model = AutoModel.from_pretrained(model_id, **kwargs)
    return model


def get_hf_model_and_processor(
    model_id: str,
    model_type: HFModelType,
    *,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
    use_fast: bool = True,
    subfolder: str = "",
    processor_subfolder: str | None = None,
) -> "tuple[PreTrainedModel, ProcessorMixin]":
    """Load and cache a Hub checkpoint with the processor that feeds it.

    ``model_type`` pairs a model class with a processor class, which fixes the
    model's forward output. Loaded weights are cached on the host. Each call
    receives an independent evaluation-mode model on
    ``caresmcp.constants.DEVICE``, while the processor is returned from the
    cache unchanged.

    Args:
        model_id: Hub repository ID of the checkpoint.
        model_type: Model and processor pairing to load. Each
            :class:`HFModelType` member fixes both classes and, in turn, the
            model output:

            * ``IMAGE_CLASSIFICATION``:
              :class:`~transformers.AutoModelForImageClassification` with
              :class:`~transformers.AutoImageProcessor`; returns an
              :class:`~transformers.modeling_outputs.ImageClassifierOutput`
              whose ``logits`` hold one score per class.
            * ``IMAGE_ENCODER``: :class:`~transformers.AutoModel` with
              :class:`~transformers.AutoImageProcessor`; encoders typically
              return a
              :class:`~transformers.modeling_outputs.BaseModelOutputWithPooling`
              with per-token states in ``last_hidden_state`` and, when the
              architecture has one, an image-level vector in ``pooler_output``.
            * ``IMAGE_PRETRAINING``:
              :class:`~transformers.AutoModelForPreTraining` with
              :class:`~transformers.AutoImageProcessor`; keeps the
              checkpoint's pretraining head, so the output is head-specific.
              For example, :class:`~transformers.ViTMAEForPreTraining` returns
              reconstruction ``logits`` and the ``mask`` of selected patches.
            * ``AUTO``: :class:`~transformers.AutoModel` with
              :class:`~transformers.AutoProcessor`; use for checkpoints with a
              custom model interface, such as vision-language dual encoders
              exposing ``get_image_features`` and ``get_text_features``.
            * ``SAM``: :class:`~transformers.SamModel` with
              :class:`~transformers.SamProcessor`; prompted image segmentation
              returning ``pred_masks`` (mask logits) and ``iou_scores``.
            * ``SAM2``: :class:`~transformers.Sam2Model` with
              :class:`~transformers.Sam2Processor`; loads only the image branch
              of a SAM2 video checkpoint, ignoring video-only weights, and
              returns ``pred_masks`` and ``iou_scores`` for one or more
              candidate masks.
            * ``SEMANTIC_SEGMENTATION``:
              :class:`~transformers.AutoModelForSemanticSegmentation` with
              :class:`~transformers.AutoImageProcessor`; returns a
              :class:`~transformers.modeling_outputs.SemanticSegmenterOutput`
              whose ``logits`` hold per-class scores at a reduced resolution
              that callers must resize to the input image.
        trust_remote_code: Load the model and processor implementations
            shipped in the repository instead of those in the installed
            ``transformers``.
        torch_dtype: Weight dtype passed to the model's ``from_pretrained``,
            for example ``"auto"`` or ``"bfloat16"``. ``None`` keeps the
            checkpoint dtype.
        use_fast: Use fast image processors when the checkpoint provides one.
            Applies only to the image-processor pairings; the slow class is
            used when no fast implementation exists. Ignored for ``SAM``,
            ``SAM2``, and ``AUTO``, which pair fixed or automatic processors.
        subfolder: Repository subfolder that contains the model weights. The
            processor is loaded from the same subfolder unless
            ``processor_subfolder`` overrides it.
        processor_subfolder: Repository subfolder that contains the processor.
            Defaults to ``subfolder``; pass an empty string to load the
            processor from the repository root.

    Returns:
        The model on ``caresmcp.constants.DEVICE`` in evaluation mode and its
        paired processor.
    """
    model, processor = _get_hf_model_and_processor(
        model_id=model_id,
        model_type=model_type,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        use_fast=use_fast,
        subfolder=subfolder,
        processor_subfolder=processor_subfolder,
    )
    return device_copy(model), processor


@lru_cache(maxsize=LRU_MAX_SIZE)
def _get_hf_model_and_processor(
    model_id: str,
    model_type: HFModelType = HFModelType.IMAGE_CLASSIFICATION,
    *,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
    use_fast: bool = True,
    subfolder: str = "",
    processor_subfolder: str | None = None,
) -> "tuple[PreTrainedModel, ProcessorMixin]":
    """Load and cache the selected model and processor pair on CPU."""
    model_kwargs: dict[str, Any] = {}
    processor_kwargs: dict[str, Any] = {}

    match model_type:
        case HFModelType.SAM:
            from transformers import SamModel, SamProcessor

            model_class = SamModel
            processor_class = SamProcessor
        case HFModelType.SAM2:
            from transformers import Sam2Model, Sam2Processor

            model_class = Sam2Model
            processor_class = Sam2Processor
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
        case HFModelType.IMAGE_PRETRAINING:
            from transformers import AutoImageProcessor, AutoModelForPreTraining

            model_class = AutoModelForPreTraining
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

    if model_type in (
        HFModelType.IMAGE_CLASSIFICATION,
        HFModelType.IMAGE_ENCODER,
        HFModelType.IMAGE_PRETRAINING,
        HFModelType.SEMANTIC_SEGMENTATION,
    ):
        # transformers warns when `use_fast` is unset for checkpoints saved with
        # a slow image processor, and fast processors become the default in an
        # upcoming release. Opt in explicitly; transformers still falls back to
        # the slow class when a checkpoint ships no fast implementation.
        processor_kwargs["use_fast"] = use_fast
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
    model = model_class.from_pretrained(model_id, **model_kwargs)
    return model, processor


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_hf_checkpoint(
    model_id: str,
    filename: str,
    *,
    revision: str | None = None,
    weights_only: bool = True,
) -> Any:
    """Download and cache a PyTorch checkpoint from a Hub repository.

    Args:
        model_id: Hub repository ID that contains the checkpoint.
        filename: Path of the checkpoint file within the repository.
        revision: Git revision, tag, or branch of the repository. ``None``
            selects the default branch.
        weights_only: Restrict :func:`torch.load` to tensors and primitive
            types. Keep the default unless the file contains other objects.

    Returns:
        The deserialized checkpoint on CPU.
    """
    import torch
    from huggingface_hub import hf_hub_download

    artifact = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        revision=revision,
    )
    # safetensors handled in torch.load()
    return torch.load(artifact, map_location="cpu", weights_only=weights_only)


@lru_cache(maxsize=LRU_MAX_SIZE)
def get_hf_json(model_id: str, filename: str, *, subfolder: str = "") -> Any:
    """Download and cache a JSON metadata file from a Hub repository.

    Args:
        model_id: Hub repository ID that contains the file.
        filename: Path of the JSON file within the repository.
        subfolder: Repository subfolder that contains the file. An empty
            string selects the repository root.

    Returns:
        The parsed JSON document.
    """
    from huggingface_hub import hf_hub_download

    artifact = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        subfolder=subfolder or None,
    )
    return json.loads(Path(artifact).read_text())
