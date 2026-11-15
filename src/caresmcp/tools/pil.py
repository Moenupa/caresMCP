"""Helpers for inspecting and preparing 2D medical images."""

from typing import Annotated, Literal

from pydantic import Field

from caresmcp.api.io import check_bbox, decode_as, encode_as
from caresmcp.registry import IMAGE_PROC


@IMAGE_PROC.register("download")
def download(
    image: bytes, convert_as: Literal["RGB", "RGBA", "L"] | None = None
) -> bytes:
    """Download and return an image."""
    if convert_as is None:
        return image

    return encode_as(decode_as(image, convert_as))


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
    bbox: Annotated[
        tuple[int, int, int, int],
        Field(description="Crop box in image pixels as (x_min, y_min, x_max, y_max)."),
    ],
) -> bytes:
    """Crop a 2D image to a pixel box."""
    pil_image = decode_as(image)
    x_min, y_min, x_max, y_max = check_bbox(bbox, *pil_image.size)
    cropped = pil_image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    return encode_as(cropped)


@IMAGE_PROC.register("cropcenter")
def cropcenter(
    image: bytes,
    width: Annotated[int, Field(gt=0)],
    height: Annotated[int, Field(gt=0)],
) -> bytes:
    """Crop the center of a 2D image to the given pixel size."""
    pil_image = decode_as(image)
    left = (pil_image.width - width) // 2
    top = (pil_image.height - height) // 2
    x_min, y_min, x_max, y_max = check_bbox(
        (left, top, left + width, top + height), *pil_image.size
    )
    cropped = pil_image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    return encode_as(cropped)
