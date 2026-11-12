"""Multi-mask musculoskeletal radiograph reconstruction error with SKELEX."""

from typing import Annotated, cast

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as, encode_as
from caresmcp.api.models import HFModelType, get_hf_model_and_processor
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_EVAL, XRAY


def _prepare_skelex_image(image: bytes) -> Image.Image:
    """Apply the upstream image-wise min/max conversion, retaining 16-bit contrast."""
    source = ImageOps.exif_transpose(decode_as(image))
    if source.mode == "P":
        source = source.convert("RGB")
    pixels = np.asarray(source)
    if pixels.ndim == 3 and pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    pixels = pixels.astype(np.float32)
    low, high = float(pixels.min()), float(pixels.max())
    scaled = (
        (pixels - low) / (high - low) * 255 if high > low else np.zeros_like(pixels)
    )
    return Image.fromarray(scaled.clip(0, 255).astype(np.uint8)).convert("RGB")


@XRAY.register("skelex_reconstruction_error")
@TASK_EVAL.register("skelex_reconstruction_error")
@requires(["torch", "transformers"], extra="xray")
def skelex_reconstruction_error(
    image: Annotated[bytes, Field(description="Musculoskeletal radiograph image")],
    passes: Annotated[
        int,
        Field(
            ge=1, le=32, description="Number of random 75%-masked reconstruction passes"
        ),
    ] = 10,
    seed: Annotated[
        int, Field(ge=0, le=2**32 - 1, description="Seed for reproducible masking")
    ] = 0,
) -> bytes:
    """Map reconstruction error in a musculoskeletal radiograph.

    Use the map to explore regions the model reconstructs poorly. Increase the
    number of passes for more coverage and set the seed to reproduce a map.
    Brighter pixels indicate larger errors, scaled to each image, so brightness
    cannot be compared across images. Transparent pixels have no error estimate.
    The 224 x 224 map stretches non-square images; PNG metadata records the error
    scale, observed-pixel mean error, coverage, passes and seed.

    Reconstruction error is not a calibrated disease score or lesion boundary.
    Weights are licensed CC-BY-NC-ND-4.0 for non-commercial use.

    See Also:
        https://github.com/skhoha/SKELEX/blob/main/unsupervised_heatmap.py
        https://huggingface.co/skhoha/SKELEX
    """
    import torch
    from transformers import ViTImageProcessor, ViTMAEForPreTraining

    model, processor = get_hf_model_and_processor(
        "skhoha/SKELEX", HFModelType.IMAGE_PRETRAINING, use_fast=False
    )
    model = cast(ViTMAEForPreTraining, model)
    processor = cast(ViTImageProcessor, processor)
    inputs = processor(images=_prepare_skelex_image(image), return_tensors="pt")
    pixels = inputs["pixel_values"].to(DEVICE)
    patch_size = model.config.patch_size
    patch_count = (pixels.shape[-2] // patch_size) * (pixels.shape[-1] // patch_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    std = torch.tensor(processor.image_std, device=DEVICE)[None, :, None, None]
    error_sum = torch.zeros_like(pixels[0, 0])
    hits = torch.zeros_like(error_sum)
    with torch.inference_mode():
        for _ in range(passes):
            noise = torch.rand((1, patch_count), generator=generator).to(DEVICE)
            output = model(pixel_values=pixels, noise=noise)
            reconstruction = model.unpatchify(output.logits)
            # Subtraction cancels the processor mean; std restores RGB units.
            error = ((reconstruction - pixels) * std).square().mean(dim=1)[0]
            mask = model.unpatchify(
                output.mask[..., None].expand(-1, -1, patch_size**2 * 3)
            )[0, 0]
            error_sum += error * mask
            hits += mask
    observed = hits > 0
    error = error_sum / hits.clamp_min(1)
    maximum = float(error.max().item())
    gray = (error / maximum * 255 if maximum > 0 else error).clamp(0, 255)
    gray_array = gray.to(torch.uint8).cpu().numpy()
    alpha = observed.to(torch.uint8).cpu().numpy() * 255
    rgba = np.stack([gray_array, gray_array, gray_array, alpha], axis=-1)
    metadata = PngImagePlugin.PngInfo()
    for name, value in {
        "model_id": "skhoha/SKELEX",
        "passes": passes,
        "seed": seed,
        "mask_ratio": model.config.mask_ratio,
        "maximum_mean_squared_error": maximum,
        "observed_pixel_mean_squared_error": float(error[observed].mean().item()),
        "observed_pixel_fraction": float(observed.float().mean().item()),
        "geometry": "released image processor canvas; not original-image coordinates",
    }.items():
        metadata.add_text(name, str(value))
    return encode_as(Image.fromarray(rgba), pnginfo=metadata)
