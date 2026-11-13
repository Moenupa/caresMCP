from functools import partial
from io import BytesIO
from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as, encode_as, is_grayscale
from caresmcp.registry import IMAGE_PROC, TASK_DEN

DenoiseMethod = Literal[
    "bilateral",
    "gaussian",
    "median",
    "nl_means",
    "tv_bregman",
    "tv_chambolle",
    "wavelet",
]


@IMAGE_PROC.register("denoise")
@TASK_DEN.register("denoise")
@requires(["skimage", "pywt"], extra="skimage")
def denoise(
    image: Annotated[bytes, Field(description="Grayscale or RGB raster image.")],
    method: DenoiseMethod = "bilateral",
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
    invariant: Annotated[
        bool, Field(description="Apply J-invariant self-supervised denoising.")
    ] = False,
) -> bytes:
    """Denoise a grayscale or RGB image using a classical filter.

    Grayscale output is a floating-point TIFF preserving the input value scale;
    RGB output is an 8-bit PNG.
    """
    from skimage.filters import gaussian, median
    from skimage.restoration import (
        denoise_bilateral,
        denoise_invariant,
        denoise_nl_means,
        denoise_tv_bregman,
        denoise_tv_chambolle,
        denoise_wavelet,
    )

    source = decode_as(image)
    grayscale = is_grayscale(source.mode)
    if not grayscale and source.mode != "RGB":
        raise ValueError("image must be grayscale or RGB")

    pixels = np.asarray(source, dtype=np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError("image must contain only finite pixel values")
    low = float(pixels.min())
    high = float(pixels.max())
    scale = high - low or 1.0
    normalized = (pixels - low) / scale
    channel_axis = None if grayscale else -1
    if method == "bilateral":
        filter_image = partial(
            denoise_bilateral,
            win_size=kernel_size,
            sigma_color=sigma_intensity / scale,
            sigma_spatial=sigma_pixels,
            channel_axis=channel_axis,
        )
    elif method == "gaussian":
        filter_image = partial(gaussian, sigma=sigma_pixels, channel_axis=channel_axis)
    elif method == "median":
        footprint = np.ones(
            (kernel_size, kernel_size) if grayscale else (kernel_size, kernel_size, 1),
            dtype=bool,
        )
        filter_image = partial(median, footprint=footprint)
    elif method == "nl_means":
        filter_image = partial(denoise_nl_means, channel_axis=channel_axis)
    elif method == "tv_bregman":
        filter_image = partial(denoise_tv_bregman, channel_axis=channel_axis)
    elif method == "tv_chambolle":
        filter_image = partial(denoise_tv_chambolle, channel_axis=channel_axis)
    else:
        filter_image = partial(denoise_wavelet, channel_axis=channel_axis)
    filtered = (
        denoise_invariant(normalized, filter_image)
        if invariant
        else filter_image(normalized)
    ) * scale + low
    if not grayscale:
        return encode_as(
            Image.fromarray(np.rint(np.clip(filtered, 0, 255)).astype(np.uint8))
        )
    output = BytesIO()
    Image.fromarray(np.asarray(filtered, dtype=np.float32)).save(output, format="TIFF")
    return output.getvalue()
