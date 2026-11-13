"""Three-class benign, malignant, and normal scoring for mammogram patches."""

from typing import Annotated, Any

from pydantic import Field

from caresmcp.api.decorators import adapt_inputs, requires
from caresmcp.api.io import decode_as, get_classification_scores
from caresmcp.api.models.umair import get_umair_classifier
from caresmcp.constants import DEVICE
from caresmcp.registry import MAMMOGRAPHY, TASK_CLS


def _prepare_patch(image: bytes) -> Any:
    """Apply the release's 224-square resize and ImageNet normalization."""
    from torchvision.transforms import v2 as transforms

    classifier = get_umair_classifier("mammography")
    preprocessing = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(classifier.mean, classifier.std),
        ]
    )
    return preprocessing(decode_as(image, "RGB")).unsqueeze(0).to(DEVICE)


@MAMMOGRAPHY.register("classify_mammogram_patch")
@TASK_CLS.register("classify_mammogram_patch")
@requires(["torch", "torchvision", "huggingface_hub"], extra="hf")
@adapt_inputs(image=_prepare_patch)
def classify_mammogram_patch(
    image: Annotated[bytes, Field(description="Mammogram image or lesion patch")],
) -> dict[str, float]:
    """Score benign, malignant, and normal classes for a mammogram patch.

    The checkpoint was trained on CLAHE-preprocessed INbreast, MIAS, and DDSM
    images. Its reported test accuracy has not been independently validated here,
    and these scores are not a screening-exam or clinical diagnosis.

    See Also:
        https://huggingface.co/umairinayat/medical-models
        https://huggingface.co/umairinayat/medical-models/blob/main/disease_models/mammography/classification/metrics.json
    """
    import torch

    classifier = get_umair_classifier("mammography")
    with torch.inference_mode():
        logits = classifier.model(image)
    return get_classification_scores(
        logits, dict(enumerate(classifier.class_names)), "softmax"
    )
