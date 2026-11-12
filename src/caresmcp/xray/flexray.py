"""Anatomical segmentation across radiographic regions with FleXray."""

from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as, encode_as
from caresmcp.api.models.flexray import get_flexray_model
from caresmcp.registry import TASK_SEG, XRAY

AnatomicalRegion = Literal[
    "skull",
    "scapulae",
    "clavicles",
    "humeri",
    "phalanges",
    "metacarpals",
    "carpals",
    "radii",
    "ulnae",
    "tibiae",
    "femurs",
    "patellae",
    "fibulae",
    "metatarsals",
    "tarsals",
    "toes",
    "rib_1",
    "rib_2",
    "rib_3",
    "rib_4",
    "rib_5",
    "rib_6",
    "rib_7",
    "rib_8",
    "rib_9",
    "rib_10",
    "rib_11",
    "rib_12",
    "sternum",
    "vertebra_c1",
    "vertebra_c2",
    "vertebra_c3",
    "vertebra_c4",
    "vertebra_c5",
    "vertebra_c6",
    "vertebra_c7",
    "vertebra_t1",
    "vertebra_t2",
    "vertebra_t3",
    "vertebra_t4",
    "vertebra_t5",
    "vertebra_t6",
    "vertebra_t7",
    "vertebra_t8",
    "vertebra_t9",
    "vertebra_t10",
    "vertebra_t11",
    "vertebra_t12",
    "vertebra_l1",
    "vertebra_l2",
    "vertebra_l3",
    "vertebra_l4",
    "vertebra_l5",
    "hips",
    "sacrum",
    "kidneys",
    "liver",
    "spleen",
    "lungs",
    "heart",
]


@XRAY.register("flexray_segment_anatomy")
@TASK_SEG.register("flexray_segment_anatomy")
@requires(["torch", "fxr"], extra="xray")
def flexray_segment_anatomy(
    image: Annotated[bytes, Field(description="Radiograph image")],
    region: Annotated[
        AnatomicalRegion, Field(description="Anatomical structure to segment")
    ],
    probability_threshold: Annotated[
        float, Field(ge=0, le=1, description="Minimum independent sigmoid pixel score")
    ] = 0.5,
) -> bytes:
    """Segment an anatomical structure in a radiograph.

    Select a region to obtain a binary mask for visual review or further image
    analysis. Masks match the input dimensions, use white for the structure and
    black for background, and can overlap across separate calls. Paired bones are
    not separated by laterality.

    See Also:
        https://huggingface.co/VictorButoi/flexray
    """
    pixels = decode_as(image)
    model = get_flexray_model()
    prediction = model.predict(pixels, label=region, threshold=probability_threshold)
    mask = prediction.masks[0, 0].cpu().numpy().astype(np.uint8) * 255
    # Invert square padding before cropping; nearest sampling retains binary IDs.
    width, height = pixels.size
    side = max(width, height)
    canvas = Image.fromarray(mask).resize((side, side), Image.Resampling.NEAREST)
    left = (side - width) // 2
    top = (side - height) // 2
    return encode_as(canvas.crop((left, top, left + width, top + height)))
