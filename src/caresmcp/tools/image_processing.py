"""Helpers for inspecting and preparing 2D medical images."""

from io import BytesIO

from PIL import Image

from caresmcp.registry import IMAGE_PROC


def decode_rgb(image: bytes) -> Image.Image:
    """Decode image bytes into an RGB PIL image."""
    with Image.open(BytesIO(image)) as pil_image:
        return pil_image.convert("RGB")


@IMAGE_PROC.register("inspect")
def inspect(image: bytes, **_) -> dict[str, object]:
    """Get metadata for a 2D image."""
    with Image.open(BytesIO(image)) as pil_image:
        return {
            "format": pil_image.format,
            "width": pil_image.width,
            "height": pil_image.height,
            "mode": pil_image.mode,
        }


@IMAGE_PROC.register("crop")
def crop(
    image: bytes,
    left: int,
    top: int,
    right: int,
    bottom: int,
    **_,
) -> bytes:
    """Crop a 2D image using pixel coordinates."""
    if right <= left or bottom <= top:
        raise ValueError("Crop bounds must satisfy right > left and bottom > top")
    with Image.open(BytesIO(image)) as pil_image:
        inside_image = (
            0 <= left < right <= pil_image.width
            and 0 <= top < bottom <= pil_image.height
        )
        if not inside_image:
            raise ValueError(
                "Crop must be inside image bounds "
                f"(0, 0, {pil_image.width}, {pil_image.height})"
            )
        cropped = pil_image.crop((left, top, right, bottom))
        with BytesIO() as output_bytes:
            cropped.save(output_bytes, format="PNG")
            return output_bytes.getvalue()


@IMAGE_PROC.register("centercrop")
def centercrop(
    image: bytes,
    width: int,
    height: int,
    **_,
) -> bytes:
    """Crop the center of a 2D image using pixel coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Crop bounds must have positive width and height")
    with Image.open(BytesIO(image)) as pil_image:
        left = (pil_image.width - width) // 2
        top = (pil_image.height - height) // 2
        if left < 0 or top < 0:
            raise ValueError(
                "Crop must be inside image bounds "
                f"(0, 0, {pil_image.width}, {pil_image.height})"
            )
        cropped = pil_image.crop((left, top, left + width, top + height))
        with BytesIO() as output_bytes:
            cropped.save(output_bytes, format="PNG")
            return output_bytes.getvalue()
