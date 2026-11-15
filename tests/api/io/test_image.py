"""Test decoding image bytes into detached, fully loaded images."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from caresmcp.api.io import decode_as


def _image_bytes(
    image: Image.Image, *, image_format: str = "PNG", **save_kwargs: object
) -> bytes:
    """Encode a generated image for decoding tests."""
    output = BytesIO()
    image.save(output, format=image_format, **save_kwargs)
    return output.getvalue()


def test_decodes_pixels_and_detaches_from_buffer():
    """Return exact pixels without retaining the source byte buffer."""
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    image = decode_as(_image_bytes(Image.fromarray(pixels)))
    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (4, 4)
    assert getattr(image, "fp", None) is None
    np.testing.assert_array_equal(np.asarray(image), pixels)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_preserves_source_format(image_format: str):
    """Keep the format recorded for the encoded input."""
    payload = _image_bytes(Image.new("RGB", (4, 4), "red"), image_format=image_format)
    assert decode_as(payload).format == image_format


def test_converts_mode_and_keeps_format():
    """Convert pixels while retaining the source format."""
    pixels = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    payload = _image_bytes(Image.fromarray(pixels))
    image = decode_as(payload, mode="L")
    with Image.open(BytesIO(payload)) as source:
        expected = np.asarray(source.convert("L"))
    assert image.mode == "L"
    assert image.format == "PNG"
    np.testing.assert_array_equal(np.asarray(image), expected)


def test_preserves_palette_and_transparency():
    """Retain palette indices, colors, and PNG transparency metadata."""
    palette = Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).convert("P")
    colors = [0, 0, 0, 255, 0, 0]
    palette.putpalette(colors + [0] * (768 - len(colors)))
    image = decode_as(_image_bytes(palette, transparency=0))
    palette_values = image.getpalette()
    assert image.mode == "P"
    assert image.info["transparency"] == 0
    assert palette_values is not None
    assert palette_values[:6] == colors
    np.testing.assert_array_equal(np.asarray(image), np.array([[0, 1], [1, 0]]))


def test_rejects_invalid_bytes():
    """Fail at decode time for data that is not an image."""
    with pytest.raises(UnidentifiedImageError):
        decode_as(b"not an image")
