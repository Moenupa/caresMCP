"""MedSAM promptable segmentation with the official inference pipeline."""

import numpy as np
from PIL import Image

from caresmcp.api.io import check_bbox
from caresmcp.api.models import HFModelType, get_hf_model_and_processor
from caresmcp.constants import DEVICE

MEDSAM_MODEL_ID = "flaviagiammarino/medsam-vit-base"


def medsam_mask(
    source: Image.Image,
    box: tuple[float, float, float, float],
    model_id: str,
) -> np.ndarray:
    """Return an original-size binary mask through the official MedSAM pipeline.

    Args:
        source: RGB image containing the prompted region.
        box: Prompt box in original pixels as ``(x1, y1, x2, y2)``.
        model_id: Hub repository ID of a MedSAM checkpoint.

    Returns:
        A boolean foreground mask at the source image size.
    """
    import torch
    from torch.nn import functional as F
    from torchvision.transforms.functional import InterpolationMode, resize

    width, height = source.size
    x1, y1, x2, y2 = check_bbox(box, width, height)
    model, _ = get_hf_model_and_processor(model_id, HFModelType.SAM)
    pixels = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)
    pixels = resize(
        pixels,
        [1024, 1024],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    resized = pixels.permute(1, 2, 0).numpy().astype(np.float32)
    resized = (resized - resized.min()) / np.clip(
        resized.max() - resized.min(), a_min=1e-8, a_max=None
    )
    pixel_values = (
        torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    )
    input_boxes = (
        torch.tensor([[x1, y1, x2, y2]], dtype=torch.float, device=DEVICE)
        / torch.tensor([width, height, width, height], device=DEVICE)
        * 1024
    )
    with torch.inference_mode():
        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes[:, None, :],
            multimask_output=False,
        )
    logits = outputs.pred_masks.sigmoid()[:, 0]
    mask = (
        (
            F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0]
            > 0.5
        )
        .cpu()
        .numpy()
    )
    if mask.shape != (height, width) or not mask.any():
        raise RuntimeError("Segmenter returned an empty or mis-sized mask")
    return mask
