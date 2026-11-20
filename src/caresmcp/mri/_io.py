"""Shared reconstructed-slice adapters; no acquisition geometry is inferred."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from caresmcp.api.io import (
    decode_single_frame,
    float_slice,
    grayscale_array,
    normalize_minmax,
)


def _decode_mri_slice(
    image: bytes, *, normalize: bool = True
) -> tuple[NDArray[np.float32], float, float, Any]:
    """Decode a grayscale MRI slice and normalize its finite intensity range.

    Args:
        image: Encoded single-frame raster image.
        normalize: Min-max normalize to ``[0, 1]``. When False, return the
            native intensities as ``float32``.

    Returns:
        The (possibly normalized) pixels, the source minimum, the source
        maximum, and the decoded NumPy dtype.
    """
    source = decode_single_frame(image)
    pixels = grayscale_array(source, convert=True)
    source_dtype = pixels.dtype
    if not normalize:
        values = float_slice(pixels)
        return values, float(values.min()), float(values.max()), source_dtype
    normalized, minimum, maximum = normalize_minmax(pixels)
    return normalized, minimum, maximum, source_dtype


def _decode_float_mri_stack(
    images: Sequence[bytes], description: str
) -> list[NDArray[np.float32]]:
    """Decode same-sized float TIFF slices while preserving native values.

    Args:
        images: Encoded single-frame slices in acquisition order.
        description: Input group named in validation errors.

    Returns:
        The native-value ``float32`` slices.

    Raises:
        ValueError: If a slice is not a float TIFF or the sizes differ.
    """
    decoded = [_decode_mri_slice(image, normalize=False) for image in images]
    if any(not np.issubdtype(item[3], np.floating) for item in decoded):
        raise ValueError(
            f"{description} must be float TIFFs preserving native intensities"
        )
    slices = [item[0] for item in decoded]
    if any(image.shape != slices[0].shape for image in slices):
        raise ValueError(f"{description} must have identical dimensions")
    return slices
