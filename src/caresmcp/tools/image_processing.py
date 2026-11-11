"""Helpers for inspecting and preparing 2D medical images."""

from io import BytesIO
from typing import Annotated, Literal, Self

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from caresmcp.api.io import decode_as, encode_as, is_grayscale
from caresmcp.registry import IMAGE_PROC, TASK_DEN


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


class _DenoiseOptions(BaseModel):
    method: Literal["bilateral", "gaussian", "median"]
    kernel_size: Literal[3, 5, 7, 9, 11, 13, 15]

    @model_validator(mode="after")
    def validate_median_kernel(self) -> Self:
        if self.method == "median" and self.kernel_size > 5:
            raise ValueError("median filtering supports kernel_size 3 or 5")
        return self


@IMAGE_PROC.register("denoise")
@TASK_DEN.register("denoise")
def denoise(
    image: Annotated[bytes, Field(description="Grayscale or RGB raster image.")],
    method: Literal["bilateral", "gaussian", "median"] = "bilateral",
    kernel_size: Literal[3, 5, 7, 9, 11, 13, 15] = 5,
    sigma_intensity: Annotated[
        float,
        Field(
            gt=0,
            allow_inf_nan=False,
            description="Bilateral intensity scale in image value units.",
        ),
    ] = 50.0,
    sigma_pixels: Annotated[
        float,
        Field(
            gt=0,
            allow_inf_nan=False,
            description="Bilateral spatial scale or Gaussian standard deviation in pixels.",
        ),
    ] = 1.5,
) -> bytes:
    """Denoise a grayscale or RGB image using a classical filter.

    Grayscale output is a floating-point TIFF preserving the input value scale;
    RGB output is an 8-bit PNG.
    """
    _DenoiseOptions(method=method, kernel_size=kernel_size)
    source = decode_as(image)
    grayscale = is_grayscale(source.mode)
    if not grayscale and source.mode != "RGB":
        raise ValueError("image must be grayscale or RGB")
    pixels = np.asarray(source, dtype=np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError("image must contain only finite pixel values")
    if method == "bilateral":
        filtered = cv2.bilateralFilter(
            pixels, kernel_size, sigmaColor=sigma_intensity, sigmaSpace=sigma_pixels
        )
    elif method == "gaussian":
        filtered = cv2.GaussianBlur(
            pixels, (kernel_size, kernel_size), sigmaX=sigma_pixels
        )
    else:
        filtered = cv2.medianBlur(pixels, kernel_size)
    if not grayscale:
        return encode_as(
            Image.fromarray(np.rint(np.clip(filtered, 0, 255)).astype(np.uint8))
        )
    output = BytesIO()
    Image.fromarray(np.asarray(filtered, dtype=np.float32)).save(output, format="TIFF")
    return output.getvalue()
