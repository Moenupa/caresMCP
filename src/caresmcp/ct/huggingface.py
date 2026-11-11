"""CT slice classification with supported Hub checkpoints."""

from typing import Any

from caresmcp.api.decorators import requires
from caresmcp.api.io import get_classification_scores
from caresmcp.api.models.hf_hub import (
    HFModelType,
    get_hf_checkpoint,
    get_hf_model_and_processor,
)
from caresmcp.api.models.timm import get_timm_model
from caresmcp.constants import DEVICE
from caresmcp.registry import CT, TASK_CLS

from ._arg import (
    ClassifyCtSliceModelIdArg,
    CTImage,
    RasterRescaleIntercept,
    RasterRescaleSlope,
    RasterUnits,
    WindowCenter,
    WindowWidth,
)
from ._arg import CTClassifier as CTClassifier  # noqa: PLC0414
from .io import (
    CTRasterUnits,
    _decode_ct_slice,
    _model_image,
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


@CT.register("classify_ct_slice")
@TASK_CLS.register("classify_ct_slice")
@requires(["torch", "transformers"], extra="hf")
def classify_ct_slice(
    image_or_dicom: CTImage,
    model_id: ClassifyCtSliceModelIdArg = "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans",
    rescale_slope: RasterRescaleSlope = 1.0,
    rescale_intercept: RasterRescaleIntercept = 0.0,
    window_center: WindowCenter = 40.0,
    window_width: WindowWidth = 400.0,
    raster_units: RasterUnits = "display",
) -> dict[str, Any]:
    """Return classification scores for one CT slice.

    See Also:
        https://huggingface.co/oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans
        https://huggingface.co/sukhmani1303/lung-cancer-vit-model
    """
    import torch

    decoded = _decode_ct_slice(
        image_or_dicom, rescale_slope, rescale_intercept, raster_units
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
    scores = get_classification_scores(logits, labels, "softmax")
    return {
        "model_id": model_id,
        "task": "lung_cancer_classification",
        "score_mode": "softmax",
        **_window_metadata(decoded.intensity_units, window_center, window_width),
        "scores": scores,
    }
