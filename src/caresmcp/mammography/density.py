"""Breast density classification with the MONAI model zoo checkpoint."""

from typing import Annotated, Any

from pydantic import Field

from caresmcp.api.decorators import adapt_inputs, requires
from caresmcp.api.io import decode_as, get_classification_scores
from caresmcp.api.models import get_monai_model
from caresmcp.constants import DEVICE
from caresmcp.registry import MAMMOGRAPHY, TAG_MONAI, TASK_CLS

_MODEL_ID = "MONAI/breast_density_classification"
_LABELS = (
    "A: almost entirely fatty",
    "B: scattered fibroglandular density",
    "C: heterogeneously dense",
    "D: extremely dense",
)


def _prepare_image(image: bytes) -> Any:
    """Apply the bundle's intensity scaling and spatial resize."""
    from monai.transforms import Compose, Resize, ScaleIntensity
    from torchvision.transforms.v2 import ToTensor

    preprocessing = Compose(
        [
            ToTensor(),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=(299, 299)),
        ]
    )
    return preprocessing(decode_as(image, "RGB")).unsqueeze(0).to(DEVICE)


@MAMMOGRAPHY.register("classify_breast_density")
@TASK_CLS.register("classify_breast_density")
@TAG_MONAI.register("classify_breast_density")
@requires(["torch", "torchvision", "monai", "huggingface_hub"], extra="monai")
@adapt_inputs(image=_prepare_image)
def classify_breast_density(
    image: Annotated[bytes, Field(description="Single mammogram image")],
) -> dict[str, float]:
    """Score the four BI-RADS breast density categories on one mammogram (in sigmoid).

    See Also:
        https://huggingface.co/MONAI/breast_density_classification
        https://arxiv.org/abs/2202.08238
    """
    import torch

    with torch.inference_mode():
        logits = get_monai_model(_MODEL_ID)(image)
        return get_classification_scores(logits, dict(enumerate(_LABELS)), "sigmoid")
