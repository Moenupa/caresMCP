"""Helpers for inspecting and preparing 2D medical images."""

from typing import Annotated

from pydantic import Field

from caresmcp.api.imageio import decode_as, encode_as
from caresmcp.registry import IMAGE_PROC


@IMAGE_PROC.register("inspect")
def inspect(image: bytes) -> dict[str, object]:
    """Get metadata for a 2D image."""
    pil_image = decode_as(image)
    return {
        "format": pil_image.format,
        "width": pil_image.width,
        "height": pil_image.height,
        "mode": pil_image.mode,
    }


@IMAGE_PROC.register("crop")
def crop(
    image: bytes,
    left: Annotated[int, Field(ge=0)],
    top: Annotated[int, Field(ge=0)],
    right: Annotated[int, Field(gt=0)],
    bottom: Annotated[int, Field(gt=0)],
) -> bytes:
    """Crop a 2D image using pixel coordinates."""
    if right <= left or bottom <= top:
        raise ValueError("Crop bounds must satisfy right > left and bottom > top")
    pil_image = decode_as(image)
    inside_image = right <= pil_image.width and bottom <= pil_image.height
    if not inside_image:
        raise ValueError(
            "Crop must be inside image bounds "
            f"(0, 0, {pil_image.width}, {pil_image.height})"
        )
    cropped = pil_image.crop((left, top, right, bottom))
    return encode_as(cropped)


@IMAGE_PROC.register("cropcenter")
def cropcenter(
    image: bytes,
    width: Annotated[int, Field(gt=0)],
    height: Annotated[int, Field(gt=0)],
) -> bytes:
    """Crop the center of a 2D image using pixel coordinates."""
    pil_image = decode_as(image)
    left = (pil_image.width - width) // 2
    top = (pil_image.height - height) // 2
    if left < 0 or top < 0:
        raise ValueError(
            "Crop must be inside image bounds "
            f"(0, 0, {pil_image.width}, {pil_image.height})"
        )
    cropped = pil_image.crop((left, top, left + width, top + height))
    return encode_as(cropped)
