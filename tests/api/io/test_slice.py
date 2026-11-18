"""Check the shared slice decoding blocks against real images."""

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import (
    decode_as,
    decode_mask,
    decode_single_frame,
    encode_as,
    float_slice,
    grayscale_array,
    normalize_minmax,
    rgb8_slice,
)

pytestmark = pytest.mark.slow

CT_IMAGE_URL = (
    "https://prod-images-static.radiopaedia.org/images/69161224/dr-original.jpg"
)
WIDE_GRAYSCALE_URL = (
    "https://raw.githubusercontent.com/python-pillow/Pillow/11.3.0/"
    "Tests/images/16bit.cropped.tif"
)
COLOR_IMAGE_URL = (
    "https://prod-images-static.radiopaedia.org/images/74884531/"
    "6f22f016116baa92e86b44a6bec617b490d615e54ebdbdc31078981193d6a675.png"
)
TRANSPARENT_IMAGE_URL = (
    "https://raw.githubusercontent.com/python-pillow/Pillow/11.3.0/"
    "Tests/images/transparent.png"
)
MULTIFRAME_URL = (
    "https://raw.githubusercontent.com/python-pillow/Pillow/11.3.0/"
    "Tests/images/iss634.gif"
)


@pytest.fixture(scope="module")
def ct_image() -> bytes:
    """A real single-frame grayscale CT slice."""
    return _read_bytes(CT_IMAGE_URL)


@pytest.fixture(scope="module")
def wide_grayscale() -> bytes:
    """A real 16-bit TIFF with intensities that overflow uint8."""
    return _read_bytes(WIDE_GRAYSCALE_URL)


@pytest.fixture(scope="module")
def color_image() -> bytes:
    """A real color pathology patch."""
    return _read_bytes(COLOR_IMAGE_URL)


@pytest.fixture(scope="module")
def transparent_image() -> bytes:
    """A real RGBA image with color hidden behind fully transparent pixels."""
    return _read_bytes(TRANSPARENT_IMAGE_URL)


@pytest.fixture(scope="module")
def multiframe_image() -> bytes:
    """A real animated GIF with 42 frames."""
    return _read_bytes(MULTIFRAME_URL)


@save_returns("slice_io", unpack=True)
def test_decode_single_frame(ct_image: bytes) -> dict[str, bytes]:
    """Forward pixels and format from the standard decode path."""
    source = decode_single_frame(ct_image)
    reference = decode_as(ct_image)
    assert (source.format, source.mode) == (reference.format, reference.mode)
    np.testing.assert_array_equal(np.asarray(source), np.asarray(reference))
    return {"decoded": encode_as(source)}


def test_decode_single_frame_rejects_multiframe(multiframe_image: bytes) -> None:
    """Reject animation containers instead of silently using frame zero."""
    with pytest.raises(ValueError, match="exactly one frame"):
        decode_single_frame(multiframe_image)


@save_returns("slice_io", unpack=True)
def test_grayscale_array(
    ct_image: bytes,
    wide_grayscale: bytes,
    color_image: bytes,
) -> dict[str, bytes]:
    """Keep grayscale dtypes and convert color only when asked."""
    source = decode_single_frame(ct_image)
    np.testing.assert_array_equal(grayscale_array(source), np.asarray(source))

    wide = grayscale_array(decode_single_frame(wide_grayscale))
    assert wide.ndim == 2
    assert wide.dtype == np.uint16
    np.testing.assert_array_equal(wide, np.asarray(decode_as(wide_grayscale)))

    color = decode_single_frame(color_image)
    with pytest.raises(ValueError, match="single-channel grayscale"):
        grayscale_array(color)
    converted = grayscale_array(color, convert=True)
    np.testing.assert_array_equal(
        converted, np.asarray(decode_as(color_image, mode="L"))
    )
    return {"converted": encode_as(Image.fromarray(converted))}


def test_float_slice_rejects_invalid_slices(wide_grayscale: bytes) -> None:
    """Reject volumes, empty slices, and non-finite pixels before arithmetic."""
    pixels = grayscale_array(decode_single_frame(wide_grayscale))
    with pytest.raises(ValueError, match="two-dimensional"):
        float_slice(pixels[None])
    with pytest.raises(ValueError, match="two-dimensional"):
        float_slice(pixels[:0])

    nonfinite = pixels.astype(np.float32)
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        float_slice(nonfinite)


@save_returns("slice_io", unpack=True)
def test_normalize_minmax(wide_grayscale: bytes) -> dict[str, bytes]:
    """Normalize wide grayscale without clipping and report the source range."""
    pixels = grayscale_array(decode_single_frame(wide_grayscale))
    values = float_slice(pixels)
    assert values.dtype == np.float32

    normalized, minimum, maximum = normalize_minmax(pixels)
    assert minimum == float(pixels.min())
    assert maximum == float(pixels.max())
    assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0
    np.testing.assert_allclose(
        normalized,
        (values - minimum) / (maximum - minimum),
    )

    constant = np.broadcast_to(pixels[:1, :1], pixels.shape)
    flat, constant_minimum, constant_maximum = normalize_minmax(constant)
    assert constant_minimum == constant_maximum == float(pixels[0, 0])
    np.testing.assert_array_equal(flat, np.zeros(pixels.shape, dtype=np.float32))

    rendered = np.rint(normalized * 255.0).astype(np.uint8)
    return {"normalized_8bit": encode_as(Image.fromarray(rendered))}


@save_returns("slice_io", unpack=True)
def test_rgb8_slice(wide_grayscale: bytes) -> dict[str, bytes]:
    """Repeat one grayscale channel into a model-ready RGB image."""
    pixels = grayscale_array(decode_single_frame(wide_grayscale))
    normalized, _, _ = normalize_minmax(pixels)
    grayscale = np.rint(normalized * 255.0).astype(np.uint8)

    rgb = rgb8_slice(grayscale)
    assert rgb.shape == (*grayscale.shape, 3)
    assert rgb.dtype == np.uint8
    for channel in range(3):
        np.testing.assert_array_equal(rgb[..., channel], grayscale)
    return {"rgb8": encode_as(Image.fromarray(rgb))}


@save_returns("slice_io", unpack=True)
def test_decode_mask(
    ct_image: bytes,
    color_image: bytes,
    transparent_image: bytes,
) -> dict[str, bytes]:
    """Treat nonzero grayscale and visible colored pixels as foreground."""
    grayscale_mask = decode_mask(ct_image)
    np.testing.assert_array_equal(grayscale_mask, np.asarray(decode_as(ct_image)) != 0)

    opaque = np.asarray(decode_as(color_image, mode="RGBA"))
    assert (opaque[..., 3] == 255).all(), "fixture must be fully opaque"
    color_mask = decode_mask(color_image)
    np.testing.assert_array_equal(color_mask, np.any(opaque[..., :3] != 0, axis=-1))

    rgba = np.asarray(decode_as(transparent_image, mode="RGBA"))
    transparent = rgba[..., 3] == 0
    hidden_color = transparent & np.any(rgba[..., :3] != 0, axis=-1)
    assert hidden_color.any(), "fixture must hide colored pixels behind zero alpha"
    transparent_mask = decode_mask(transparent_image)
    assert not transparent_mask[transparent].any()
    np.testing.assert_array_equal(
        transparent_mask[~transparent],
        np.any(rgba[..., :3] != 0, axis=-1)[~transparent],
    )

    return {
        "grayscale_mask": encode_as(
            Image.fromarray(grayscale_mask.astype(np.uint8) * 255)
        ),
        "color_mask": encode_as(Image.fromarray(color_mask.astype(np.uint8) * 255)),
        "transparent_mask": encode_as(
            Image.fromarray(transparent_mask.astype(np.uint8) * 255)
        ),
    }
