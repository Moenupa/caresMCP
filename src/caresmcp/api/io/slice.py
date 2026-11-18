"""Decode encoded raster slices into validated numeric arrays.

These helpers form the shared raster path for modality adapters. Pixel modes
and dtypes stay explicit so calibrated decoders (for example CT HU) and
uncalibrated decoders (for example MRI) can compose the same blocks without
inheriting each other's intensity assumptions.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .image import decode_as, is_grayscale


def decode_single_frame(image: bytes) -> Image.Image:
    """Decode one encoded image frame, rejecting multi-frame containers.

    Args:
        image: Encoded image bytes.

    Returns:
        The eagerly decoded single-frame image.

    Raises:
        ValueError: If the bytes are unreadable or contain multiple frames.
    """
    source = decode_as(image)
    if getattr(source, "n_frames", 1) != 1:
        raise ValueError("input must contain exactly one frame")
    return source


def grayscale_array(source: Image.Image, *, convert: bool = False) -> NDArray[Any]:
    """Return a two-dimensional grayscale array without changing its dtype.

    Args:
        source: Decoded image.
        convert: Convert a color image to 8-bit grayscale with PIL instead of
            rejecting it.

    Returns:
        The source pixels as a NumPy array.

    Raises:
        ValueError: If the image is color and ``convert`` is False.
    """
    if not is_grayscale(source.mode):
        if not convert:
            raise ValueError("input image must be single-channel grayscale")
        source = source.convert("L")
    elif source.mode == "1":
        # Use PIL's canonical 8-bit grayscale for bilevel images.
        source = source.convert("L")
    return np.asarray(source)


def float_slice(pixels: NDArray[Any]) -> NDArray[np.float32]:
    """Require one nonempty finite 2D slice and return it as float32.

    Args:
        pixels: Pixel array to validate.

    Returns:
        The pixels as ``float32``.

    Raises:
        ValueError: If the array is empty or not two-dimensional, or contains
            non-finite values.
    """
    if pixels.ndim != 2 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
        raise ValueError("slice must be a non-empty two-dimensional array")
    values = pixels.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("slice must contain only finite values")
    return values


def normalize_minmax(
    pixels: NDArray[Any],
) -> tuple[NDArray[np.float32], float, float]:
    """Normalize finite slice values to [0, 1] and return the source range.

    Args:
        pixels: Pixel array to normalize.

    Returns:
        The normalized slice, its original minimum, and its original maximum.
        A constant slice normalizes to zeros.

    Raises:
        ValueError: If the array is not a nonempty finite 2D slice.
    """
    values = float_slice(pixels)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.float32), minimum, maximum
    return (values - minimum) / (maximum - minimum), minimum, maximum


def rgb8_slice(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Repeat a two-dimensional grayscale slice into three RGB channels.

    Args:
        pixels: Two-dimensional ``uint8`` grayscale pixels.

    Returns:
        The pixels as an ``HxWx3`` ``uint8`` array.
    """
    return np.repeat(pixels[..., None], 3, axis=-1)


def decode_mask(image: bytes) -> NDArray[np.bool_]:
    """Decode a single-frame mask into a foreground boolean array.

    Grayscale masks treat nonzero pixels as foreground. Color masks require a
    nonzero alpha channel and at least one nonzero RGB channel.

    Args:
        image: Encoded mask image bytes.

    Returns:
        A two-dimensional boolean foreground mask.

    Raises:
        ValueError: If the mask is unreadable, multi-frame, or non-finite.
    """
    source = decode_single_frame(image)
    if is_grayscale(source.mode):
        values = np.asarray(source)
        if not np.isfinite(values).all():
            raise ValueError("mask must contain only finite values")
        return values != 0
    rgba = np.asarray(source.convert("RGBA"))
    return np.any(rgba[..., :3] != 0, axis=-1) & (rgba[..., 3] != 0)
