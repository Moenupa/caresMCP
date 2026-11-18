"""Test the image encoding contracts shared by the tools."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from caresmcp.api.io import decode_as, encode_as, encode_label_mask


def _pil_bytes(
    image: Image.Image, *, image_format: str = "PNG", **save_kwargs: object
) -> bytes:
    """Encode with PIL directly for formats encode_as does not support."""
    output = BytesIO()
    image.save(output, format=image_format, **save_kwargs)
    return output.getvalue()


def test_decodes_eagerly():
    """Raise at decode time for truncated input instead of leaving it lazy."""
    payload = _pil_bytes(
        Image.fromarray(np.arange(4096, dtype=np.uint8).reshape(64, 64))
    )
    # Image.open() accepts a half-truncated PNG; only load() detects the damage.
    with pytest.raises(ValueError, match="not a readable image"):
        decode_as(payload[: len(payload) // 2])


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_conversion_keeps_source_format(image_format: str):
    """Keep the format that convert() would otherwise discard."""
    pixels = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    payload = _pil_bytes(Image.fromarray(pixels), image_format=image_format)
    converted = decode_as(payload, mode="L")
    # Compare against PIL directly so the reference does not reuse decode_as.
    with Image.open(BytesIO(payload)) as source:
        expected = np.asarray(source.convert("L"))
    assert converted.mode == "L"
    assert converted.format == image_format
    np.testing.assert_array_equal(np.asarray(converted), expected)


def test_encode_as_forwards_png_save_options():
    """Forward the transparency and text options used by mask and annotation tools."""
    mask = Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).convert("P")
    mask.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("tool", "caresmcp")

    decoded = decode_as(encode_as(mask, transparency=0, pnginfo=metadata))

    assert decoded.info["transparency"] == 0
    assert decoded.info["tool"] == "caresmcp"
    np.testing.assert_array_equal(np.asarray(decoded), np.array([[0, 1], [1, 0]]))


def test_label_mask_preserves_class_indices_and_transparency():
    """Encode class IDs as palette indices with a transparent background."""
    labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    decoded = decode_as(encode_label_mask(labels))
    assert decoded.mode == "P"
    assert decoded.info["transparency"] == 0
    np.testing.assert_array_equal(np.asarray(decoded), labels)
    palette = decoded.getpalette()
    assert palette is not None
    assert palette[:12] == [
        0,
        0,
        0,
        31,
        119,
        180,
        255,
        127,
        14,
        44,
        160,
        44,
    ]
