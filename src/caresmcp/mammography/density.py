"""Breast density classification with the MONAI model zoo checkpoint."""

from typing import Annotated, Any

import numpy as np
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
    """Replicate the bundle's ``inference.json`` preprocessing exactly.

    MONAI's ``LoadImaged`` uses ``PILReader(reverse_indexing=True)``, which
    swaps the first two axes of a 2-D image, so encoding the swap here keeps
    the scores identical to a literal CPU bundle run.
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        Resize,
        ScaleIntensity,
    )

    array = np.moveaxis(np.array(decode_as(image, "RGB")), 0, 1)
    preprocessing = Compose(
        [
            EnsureChannelFirst(channel_dim=2),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=(299, 299)),
        ]
    )
    return preprocessing(array).unsqueeze(0).to(DEVICE)


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
        https://project-monai.github.io/model-zoo.html#/model/breast_density_classification

    References:
        Gupta, Vikash, et al. A multi-reconstruction study of breast density
        estimation using Deep Learning. arXiv preprint
        `arXiv:2202.08238 <https://arxiv.org/abs/2202.08238>`_ (2022).
    """
    import torch

    with torch.inference_mode():
        logits = get_monai_model(_MODEL_ID)(image)
        return get_classification_scores(logits, dict(enumerate(_LABELS)), "sigmoid")
