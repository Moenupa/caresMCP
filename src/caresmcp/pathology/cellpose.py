import math
from functools import lru_cache
from io import BytesIO
from typing import Annotated, Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from caresmcp.api.imageio import decode_rgb
from caresmcp.constants import DEVICE
from caresmcp.registry import PATHOLOGY, TASK_SEG
from caresmcp.util.moduleloader import requires

PRETRAINED_MODELS = Literal["cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"]
PATHOLOGY_TILE = Field(description="Encoded RGB or grayscale pathology tile or ROI.")
CELLPOSE_MODEL = Field(description="Pretrained Cellpose model to use.")
CELL_DIAMETER = Field(
    gt=0,
    description="Expected cell diameter in pixels, or None for automatic scale handling.",
)
FLOW_THRESHOLD = Field(
    description="Maximum flow error accepted for a segmented instance."
)
CELLPROB_THRESHOLD = Field(
    description="Cell probability cutoff used to form instances."
)
MIN_CELL_SIZE = Field(ge=0, description="Minimum instance area in pixels.")


@lru_cache(maxsize=2)
def _load_cellpose_model(pretrained_model: PRETRAINED_MODELS = "cpsam") -> Any:
    """Load and cache the pretrained Cellpose-SAM model."""
    from cellpose.models import CellposeModel

    return CellposeModel(
        gpu=DEVICE != "cpu",
        pretrained_model=pretrained_model,
    )


@PATHOLOGY.register("cellpose_segment_pathology_cells")
@TASK_SEG.register("cellpose_segment_pathology_cells")
@requires("cellpose", extra="pathology")
def cellpose_segment_pathology_cells(
    image: Annotated[bytes, PATHOLOGY_TILE],
    pretrained_model: Annotated[PRETRAINED_MODELS, CELLPOSE_MODEL] = "cpsam",
    diameter: Annotated[float | None, CELL_DIAMETER] = None,
    flow_threshold: Annotated[float, FLOW_THRESHOLD] = 0.4,
    cellprob_threshold: Annotated[float, CELLPROB_THRESHOLD] = 0.0,
    min_size: Annotated[int, MIN_CELL_SIZE] = 15,
) -> bytes:
    """Segment cells in a pathology image and show their boundaries.


    See Also:
        https://cellpose.readthedocs.io/en/latest/models.html
        https://cellpose.readthedocs.io/en/latest/outputs.html
    """

    if diameter is not None and (not math.isfinite(diameter) or diameter <= 0):
        raise ValueError("diameter must be a finite positive number or None")
    if not math.isfinite(flow_threshold):
        raise ValueError("flow_threshold must be finite")
    if not math.isfinite(cellprob_threshold):
        raise ValueError("cellprob_threshold must be finite")
    if min_size < 0:
        raise ValueError("min_size must be non-negative")

    try:
        rgb_image = decode_rgb(image)
    except (OSError, ValueError) as exc:
        raise ValueError("image must contain a readable pathology tile") from exc

    pixels = np.asarray(rgb_image, dtype=np.uint8).copy()
    model = _load_cellpose_model(pretrained_model=pretrained_model)
    result = model.eval(
        pixels,
        channel_axis=-1,
        normalize=True,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    if not isinstance(result, tuple) or not result:
        raise RuntimeError("Cellpose returned an invalid segmentation result")

    masks = np.asarray(result[0])
    if masks.ndim != 2 or masks.shape != pixels.shape[:2]:
        raise RuntimeError("Cellpose returned a mask with an unexpected shape")
    if np.any(masks < 0):
        raise RuntimeError("Cellpose returned negative instance labels")

    instance_count = int(np.unique(masks[masks > 0]).size)
    boundaries = np.zeros(masks.shape, dtype=bool)
    boundaries[1:, :] |= masks[1:, :] != masks[:-1, :]
    boundaries[:-1, :] |= masks[:-1, :] != masks[1:, :]
    boundaries[:, 1:] |= masks[:, 1:] != masks[:, :-1]
    boundaries[:, :-1] |= masks[:, :-1] != masks[:, 1:]
    boundaries &= masks > 0

    overlay = pixels.copy()
    if np.any(boundaries):
        boundary_color = np.array((255, 235, 59), dtype=np.float32)
        overlay[boundaries] = np.uint8(
            0.15 * overlay[boundaries].astype(np.float32) + 0.85 * boundary_color
        )

    output_image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(output_image)
    font = ImageFont.load_default()
    label = f"Cell instances: {instance_count}"
    text_box = draw.textbbox((0, 0), label, font=font)
    padding = 4
    box = (
        4,
        4,
        text_box[2] - text_box[0] + padding * 2 + 4,
        text_box[3] - text_box[1] + padding * 2 + 4,
    )
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text(
        (box[0] + padding, box[1] + padding),
        label,
        fill=(255, 255, 255),
        font=font,
    )

    output = BytesIO()
    output_image.save(output, format="PNG")
    return output.getvalue()
