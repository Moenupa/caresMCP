"""CT slice embeddings and classification with supported Hub checkpoints."""

import math
from typing import Annotated, Any, Literal

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.models.hf_hub import (
    HFModelType,
    get_hf_checkpoint,
    get_hf_model_and_processor,
)
from caresmcp.api.models.timm import get_timm_model
from caresmcp.api.tensorio import get_classification_scores
from caresmcp.constants import DEVICE
from caresmcp.registry import CT, TASK_CLS, TASK_EMBED

from .io import (
    CTInputFormat,
    CTRasterUnits,
    _decode_ct_slice,
    _model_image,
)

CTEncoder = Literal["facebook/dinov2-base"]
CTClassifier = Literal[
    "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans",
    "sukhmani1303/lung-cancer-vit-model",
]


CT_IMAGE = Field(
    min_length=1, description="Single-slice CT DICOM or grayscale raster image bytes."
)
CT_INPUT_FORMAT = Field(description="CT input decoding mode.")
CT_RESCALE_SLOPE = Field(
    allow_inf_nan=False, description="Stored-value multiplier for HU raster inputs."
)
CT_RESCALE_INTERCEPT = Field(
    allow_inf_nan=False, description="Stored-value offset for HU raster inputs."
)
CT_WINDOW_CENTER = Field(
    allow_inf_nan=False, description="HU center used to render calibrated HU inputs."
)
CT_WINDOW_WIDTH = Field(
    ge=1,
    allow_inf_nan=False,
    description="HU width used to render calibrated HU inputs.",
)
CT_RASTER_UNITS = Field(
    description="Display intensities (preserved) or known HU calibration. Display exports cannot recover their original HU values."
)


def _window_metadata(
    intensity_units: CTRasterUnits, center: float, width: float
) -> dict[str, Any]:
    """Describe the window applied to calibrated HU inputs."""
    return {
        "intensity_units": intensity_units,
        "window_center": center if intensity_units == "hu" else None,
        "window_width": width if intensity_units == "hu" else None,
    }


@CT.register("ct_slice_image_embed")
@TASK_EMBED.register("ct_slice_image_embed")
@requires(["torch", "transformers"], extra="hf")
def ct_slice_image_embed(
    image: Annotated[bytes, CT_IMAGE],
    model_id: Annotated[
        CTEncoder,
        Field(description="DINOv2 Base encoder trained on natural images, not CT."),
    ] = "facebook/dinov2-base",
    input_format: Annotated[CTInputFormat, CT_INPUT_FORMAT] = "auto",
    rescale_slope: Annotated[float, CT_RESCALE_SLOPE] = 1.0,
    rescale_intercept: Annotated[float, CT_RESCALE_INTERCEPT] = 0.0,
    window_center: Annotated[float, CT_WINDOW_CENTER] = 40.0,
    window_width: Annotated[float, CT_WINDOW_WIDTH] = 400.0,
    pooling: Annotated[
        Literal["cls", "mean"],
        Field(description="Use the first token or mean of non-CLS tokens."),
    ] = "cls",
    raster_units: Annotated[CTRasterUnits, CT_RASTER_UNITS] = "display",
) -> dict[str, Any]:
    """Return a feature vector for one CT slice.

    The result includes the embedding, model ID, pooling method, and intensity
    units. Window values are null for display images.

    See Also:
        https://huggingface.co/facebook/dinov2-base
    """
    import torch

    decoded = _decode_ct_slice(
        image, input_format, rescale_slope, rescale_intercept, raster_units
    )
    ct_image = _model_image(decoded, window_center, window_width)
    model, processor = get_hf_model_and_processor(model_id, HFModelType.IMAGE_ENCODER)
    inputs = processor(images=ct_image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model(**inputs)

    tokens = getattr(output, "last_hidden_state", None)
    if tokens is not None and tokens.ndim == 3 and tokens.shape[0] == 1:
        if tokens.shape[1] == 0 or tokens.shape[2] == 0:
            raise ValueError("encoder returned an empty last_hidden_state")
        if pooling == "cls" or tokens.shape[1] == 1:
            vector = tokens[:, 0, :]
            pooling_used = "cls"
        else:
            vector = tokens[:, 1:, :].mean(dim=1)
            pooling_used = "mean"
    else:
        vector = getattr(output, "pooler_output", None)
        if vector is None or vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError(
                "encoder must return a single-item last_hidden_state or pooler_output"
            )
        pooling_used = "pooler"
    embedding = vector[0].detach().float().cpu().tolist()
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("encoder returned empty or non-finite feature values")
    return {
        "model_id": model_id,
        "pooling": pooling_used,
        **_window_metadata(decoded.intensity_units, window_center, window_width),
        "embedding": embedding,
    }


@CT.register("classify_ct_slice")
@TASK_CLS.register("classify_ct_slice")
@requires(["torch", "transformers"], extra="hf")
def classify_ct_slice(
    image: Annotated[bytes, CT_IMAGE],
    model_id: Annotated[
        CTClassifier,
        Field(
            description="Swin Tiny or IQ-OTH/NCCD ViT lung CT checkpoint repository."
        ),
    ] = "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans",
    input_format: Annotated[CTInputFormat, CT_INPUT_FORMAT] = "auto",
    rescale_slope: Annotated[float, CT_RESCALE_SLOPE] = 1.0,
    rescale_intercept: Annotated[float, CT_RESCALE_INTERCEPT] = 0.0,
    window_center: Annotated[float, CT_WINDOW_CENTER] = 40.0,
    window_width: Annotated[float, CT_WINDOW_WIDTH] = 400.0,
    score_mode: Annotated[
        Literal["softmax"], Field(description="Normalize class logits with softmax.")
    ] = "softmax",
    raster_units: Annotated[CTRasterUnits, CT_RASTER_UNITS] = "display",
) -> dict[str, Any]:
    """Return classification scores for one CT slice.

    See Also:
        https://huggingface.co/oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans
        https://huggingface.co/sukhmani1303/lung-cancer-vit-model
    """
    import torch

    decoded = _decode_ct_slice(
        image, input_format, rescale_slope, rescale_intercept, raster_units
    )
    ct_image = _model_image(decoded, window_center, window_width)
    if model_id == "sukhmani1303/lung-cancer-vit-model":
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        inputs = preprocess(ct_image).unsqueeze(0).to(DEVICE)
        model = get_timm_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=2,
        )
        model.load_state_dict(
            get_hf_checkpoint(model_id, "pytorch_model.bin"), strict=True
        )
        with torch.inference_mode():
            logits = model(inputs)
        labels = {0: "Cancer", 1: "Normal"}
    else:
        model, processor = get_hf_model_and_processor(
            model_id, HFModelType.IMAGE_CLASSIFICATION
        )
        inputs = processor(images=ct_image, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            logits = model(**inputs).logits
        labels = model.config.id2label or {}
    scores = get_classification_scores(logits, labels, score_mode)
    return {
        "model_id": model_id,
        "task": "lung_cancer_classification",
        "score_mode": score_mode,
        **_window_metadata(decoded.intensity_units, window_center, window_width),
        "scores": scores,
    }
