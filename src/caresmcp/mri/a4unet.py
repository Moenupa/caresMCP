"""Four-contrast BraTS inference with the original A4-Unet architecture."""

from typing import Annotated, Any, Literal

import numpy as np

from caresmcp.api.decorators import requires
from caresmcp.api.io import encode_label_mask
from caresmcp.api.models import HFMODEL_ID_DESC, get_hf_checkpoint
from caresmcp.constants import DEVICE
from caresmcp.registry import MRI, TASK_SEG

from ._arg import MriNativeSlice
from ._io import _decode_float_mri_stack

MRI_A4UNET_CHECKPOINT = "checkpoint_epoch20.pth"

MRI_A4UNET_IMAGE_SIZE = 128
MRI_A4UNET_NUM_CLASSES = 2
MRI_A4UNET_CONFIG = {
    "image_size": MRI_A4UNET_IMAGE_SIZE,
    "num_channels": 128,
    "num_res_blocks": 2,
    "num_classes": MRI_A4UNET_NUM_CLASSES,
    "learn_sigma": True,
    "in_ch": 4,
}


def _load_a4unet(model_id: str) -> Any:
    """Load the author's four-channel, binary BraTS checkpoint strictly."""

    if DEVICE != "cuda":
        raise ValueError("the official A4-Unet implementation requires CUDA device 0")
    try:
        from a4unet.model.a4unet import (  # ty: ignore[import-not-found]  # Optional upstream package.
            create_a4unet_model,
        )
    except ImportError as exc:
        raise ImportError(
            "install RoxanneWAAANG/A4-Unet dependencies and expose its repository on PYTHONPATH in a separate worker"
        ) from exc

    model = create_a4unet_model(**dict(MRI_A4UNET_CONFIG))
    state = get_hf_checkpoint(model_id, MRI_A4UNET_CHECKPOINT)
    state = {name: value for name, value in state.items() if name != "mask_values"}
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


@MRI.register("segment_mri_a4unet")
@TASK_SEG.register("segment_mri_a4unet")
@requires(["torch", "huggingface_hub"], extra="mri")
def segment_mri_a4unet(
    t1_image: MriNativeSlice,
    t1ce_image: MriNativeSlice,
    t2_image: MriNativeSlice,
    flair_image: MriNativeSlice,
    model_id: Annotated[
        Literal["Roxanne-WANG/A4-Unet"], HFMODEL_ID_DESC
    ] = "Roxanne-WANG/A4-Unet",
) -> bytes:
    """Segment the whole-tumor region from four aligned BraTS contrasts.

    Supply native-value grayscale TIFF slices extracted from co-registered
    BraTS volumes at the same axial position and orientation. Inputs must
    preserve the author's T1, T1ce, T2, FLAIR channel ordering. Equal sizes do
    not establish registration; callers are responsible for correspondence.
    Requires the original A4-Unet source and dependencies in a CUDA worker.

    Returns:
        Original-resolution palette PNG of binary whole-tumor labels: 0 is
        transparent background and 1 is tumor. BraTS subregions are collapsed
        by the author's binary training target.

    See Also:
        https://huggingface.co/Roxanne-WANG/A4-Unet
        https://github.com/RoxanneWAAANG/A4-Unet/blob/main/predict.py
        https://github.com/RoxanneWAAANG/A4-Unet/blob/main/a4unet/dataloader/bratsloader.py
    """
    import torch
    from torch.nn import functional as F

    slices = _decode_float_mri_stack(
        (t1_image, t1ce_image, t2_image, flair_image), "all four contrasts"
    )
    tensor = torch.from_numpy(np.stack(slices))[None].to(DEVICE)
    # The official torchvision Resize default is bilinear, with antialiasing.
    tensor = F.interpolate(
        tensor,
        size=(MRI_A4UNET_IMAGE_SIZE, MRI_A4UNET_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    model = _load_a4unet(model_id)
    with torch.inference_mode():
        logits = model(tensor)
        if (
            logits.shape
            != (
                1,
                MRI_A4UNET_NUM_CLASSES,
                MRI_A4UNET_IMAGE_SIZE,
                MRI_A4UNET_IMAGE_SIZE,
            )
            or not torch.isfinite(logits).all()
        ):
            raise RuntimeError("A4-Unet returned invalid binary segmentation logits")
        # The released final-test path upsamples logits before taking argmax.
        labels = F.interpolate(
            logits,
            size=slices[0].shape,
            mode="bilinear",
            align_corners=True,
        ).argmax(1)[0]
    return encode_label_mask(labels.cpu().numpy().astype(np.uint8))
