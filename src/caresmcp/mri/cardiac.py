"""MONAI ventricular short-axis inference."""

from types import MappingProxyType
from typing import Annotated, Literal

from caresmcp.api.decorators import requires
from caresmcp.api.io import encode_label_mask
from caresmcp.api.models import HFMODEL_ID_DESC, get_monai_model
from caresmcp.constants import DEVICE
from caresmcp.registry import MRI, TAG_MONAI, TASK_SEG

from ._arg import MriSlice
from ._io import _decode_mri_slice

MRI_MONAI_CARDIAC_IMAGE_SIZE = 256
MRI_MONAI_CARDIAC_LABELS = MappingProxyType(
    {0: "background", 1: "lv_pool", 2: "myocardium", 3: "rv_pool"}
)


@MRI.register("segment_cardiac_mri_slice")
@TAG_MONAI.register("segment_cardiac_mri_slice")
@TASK_SEG.register("segment_cardiac_mri_slice")
@requires(["torch", "monai", "huggingface_hub"], extra="monai")
def segment_cardiac_mri_slice(
    image: MriSlice,
    model_id: Annotated[
        Literal["MONAI/ventricular_short_axis_3label"], HFMODEL_ID_DESC
    ] = "MONAI/ventricular_short_axis_3label",
) -> bytes:
    """Segment the LV cavity, myocardium, and RV cavity in a cardiac MR slice.

    This uses the pinned MONAI ventricular short-axis three-label model. It
    expects a 256 by 256 short-axis slice and uses min-max intensity scaling.
    It was trained predominantly on end-diastolic frames and should not be
    treated as a general-purpose MRI segmenter.

    Returns:
        A color-indexed PNG mask. Palette indices are background (0), LV
        cavity (1), myocardium (2), and RV cavity (3); index 0 is transparent.

    See Also:
        https://huggingface.co/MONAI/ventricular_short_axis_3label
    """
    import torch

    normalized, _, _, _ = _decode_mri_slice(image)
    if normalized.shape != (MRI_MONAI_CARDIAC_IMAGE_SIZE,) * 2:
        raise ValueError("cardiac MRI slice must be 256 by 256 pixels")
    model = get_monai_model(model_id)
    tensor = torch.from_numpy(normalized[None, None]).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        if (
            logits.shape
            != (
                1,
                len(MRI_MONAI_CARDIAC_LABELS),
                MRI_MONAI_CARDIAC_IMAGE_SIZE,
                MRI_MONAI_CARDIAC_IMAGE_SIZE,
            )
            or not torch.isfinite(logits).all()
        ):
            raise RuntimeError("cardiac segmenter returned invalid class logits")
        labels = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

    return encode_label_mask(labels)
