"""Helpers for inspecting and preparing 2D medical images."""

from io import BytesIO

from PIL import Image

from caresmcp.registry import IMAGE_PROC


def decode_rgb(image_obj: bytes) -> Image.Image:
    """Decode image bytes into an RGB PIL image."""
    with Image.open(BytesIO(image_obj)) as image:
        return image.convert("RGB")


@IMAGE_PROC.register("inspect")
def inspect(image_obj: bytes, **_) -> dict[str, object]:
    """Get metadata for a 2D image."""
    with Image.open(BytesIO(image_obj)) as image:
        return {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }


@IMAGE_PROC.register("crop")
def crop(
    image_obj: bytes,
    left: int,
    top: int,
    right: int,
    bottom: int,
    **_,
) -> bytes:
    """Crop a 2D image using pixel coordinates."""
    if right <= left or bottom <= top:
        raise ValueError("Crop bounds must satisfy right > left and bottom > top")
    with Image.open(BytesIO(image_obj)) as image:
        inside_image = (
            0 <= left < right <= image.width and 0 <= top < bottom <= image.height
        )
        if not inside_image:
            raise ValueError(
                "Crop must be inside image bounds "
                f"(0, 0, {image.width}, {image.height})"
            )
        cropped = image.crop((left, top, right, bottom))
        with BytesIO() as output_bytes:
            cropped.save(output_bytes, format="PNG")
            return output_bytes.getvalue()


@IMAGE_PROC.register("centercrop")
def centercrop(
    image_obj: bytes,
    width: int,
    height: int,
    **_,
) -> bytes:
    """Crop the center of a 2D image using pixel coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Crop bounds must have positive width and height")
    with Image.open(BytesIO(image_obj)) as image:
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        if left < 0 or top < 0:
            raise ValueError(
                "Crop must be inside image bounds "
                f"(0, 0, {image.width}, {image.height})"
            )
        cropped = image.crop((left, top, left + width, top + height))
        with BytesIO() as output_bytes:
            cropped.save(output_bytes, format="PNG")
            return output_bytes.getvalue()
