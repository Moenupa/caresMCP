"""Reproduce SKELEX's released error-map pipeline on its walkthrough example.

The map itself follows the released
``https://github.com/skhoha/SKELEX/blob/main/unsupervised_heatmap.py`` pipeline:
ten seeded 75%-masked reconstructions, denormalized to [0, 1] and clamped,
with the squared channel error averaged where each pixel was masked. The
published article (https://doi.org/10.1038/s41746-026-02826-9, PMCID
PMC13526867) thresholds the continuous map at its top 15% and reports Dice
against BTXRD masks above an area-matched random baseline (Fig. 3d,
P < 0.0001). That threshold cannot be applied to the tool's 8-bit output:
most low errors truncate to zero, so the top 15% would be mostly arbitrary
ties. The localization check therefore uses the smallest positive level and
tests the paper's directional claim instead of a published per-image score.

The walkthrough image is the notebook's BTXRD example, not the paper's
Fig. 3a case; the released BTXRD tumor annotations are still available for it.
"""

import io
import json
from typing import cast

import numpy as np
import pytest
from PIL import Image, ImageOps

from caresmcp.api.decorators import download_bytes, save_returns, skip_on_import_error
from caresmcp.api.io import decode_as, encode_as
from caresmcp.api.metrics import mask_overlap
from caresmcp.api.models import HFModelType, get_hf_model_and_processor
from caresmcp.constants import DEVICE
from caresmcp.xray.skelex import _prepare_skelex_image, skelex_reconstruction_error

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

# The input from the released SKELEX walkthrough notebook.
SKELEX_EXAMPLE_URL = (
    "https://drive.google.com/uc?export=download&id=1yav38aE_73z2zdZOQR87nenGG_RkRxOW"
)
SKELEX_EXAMPLE_SHA256 = (
    "f7b58cd92eae19e5263ec114b96b5bbc2a4152da4c5e96823c3dcbb5d4213bbf"
)
# Revision-pinned reformatted copy of the official BTXRD release
# (https://doi.org/10.6084/m9.figshare.27865398); its masks are rasterized
# from the released LabelMe polygons.
BTXRD_DATASET_REVISION = "5724dcf6e6b7d5244c1cc705ed05fb9c2381f4e8"
# Row zero is IMG000001, so offset 1 selects the walkthrough case IMG000002.
BTXRD_EXAMPLE_ROW_OFFSET = 1
BTXRD_MASK_SHA256 = "84ed9b71cd96bc825d138650df9a6410dade38517747e34ed0ea6b7f7f10be00"

# The paper's N=10 masking passes, with the tool's seed extension.
PASSES = 10
SEED = 0
RANDOM_MASK_TRIALS = 1000


@pytest.fixture(scope="session")
def skelex_walkthrough_example():
    """Supply SKELEX's walkthrough radiograph and its released tumor mask.

    The walkthrough image is the released BTXRD IMG000002, re-encoded by the
    walkthrough notebook; its decoded pixels are checked against the
    revision-pinned mirror that also supplies the mask. The mirror rasterizes
    the released LabelMe polygons, and the row's lesion annotations select the
    tumor class.
    """
    image = download_bytes(SKELEX_EXAMPLE_URL, checksum=SKELEX_EXAMPLE_SHA256)
    url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=MedOtter/BTXRD&config=default&split=train"
        f"&offset={BTXRD_EXAMPLE_ROW_OFFSET}&length=1"
        f"&revision={BTXRD_DATASET_REVISION}"
    )
    row = json.loads(download_bytes(url))["rows"][0]["row"]
    assert row["image_id"] == "IMG000002.jpeg", "BTXRD row order changed"
    # The mirror image and the walkthrough are the same radiograph in different
    # JPEG encodings; the mask coordinates only apply if their pixels agree.
    walkthrough = decode_as(image, mode="RGB")
    mirror = decode_as(download_bytes(row["image"]["src"]), mode="RGB")
    assert mirror.size == walkthrough.size, "BTXRD image geometry changed"
    difference = np.abs(
        np.asarray(mirror, dtype=np.int16) - np.asarray(walkthrough, dtype=np.int16)
    )
    assert difference.mean() < 1, "walkthrough radiograph differs from BTXRD row"
    mask_bytes = download_bytes(row["mask"]["src"], checksum=BTXRD_MASK_SHA256)
    shapes = json.loads(row["bboxes_json"])
    assert [(shape["label"], shape["class_id"]) for shape in shapes] == [
        ("osteosarcoma", 8)
    ], "BTXRD lesion annotations changed"
    # The mirror mask carries lossy-encoding values around the target boundary;
    # select only the class named by this image's lesion annotations.
    tumor_classes = [shape["class_id"] for shape in shapes]
    ground_truth = np.isin(np.asarray(decode_as(mask_bytes)), tumor_classes)
    assert ground_truth.any(), "BTXRD tumor mask is empty"
    return image, ground_truth, row


def _official_preprocessed_image(image: bytes) -> Image.Image:
    """Mirror ``load_rgb_8bit`` from the released ``unsupervised_heatmap.py``."""
    with Image.open(io.BytesIO(image)) as source:
        source = ImageOps.exif_transpose(source)
        pixels = np.asarray(source)
    if pixels.ndim == 3 and pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    pixels = pixels.astype(np.float32)
    low, high = np.nanmin(pixels), np.nanmax(pixels)
    if not np.isfinite([low, high]).all() or high <= low:
        pixels = np.zeros_like(pixels, dtype=np.uint8)
    else:
        pixels = np.clip((pixels - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    if pixels.ndim == 2:
        return Image.fromarray(pixels).convert("RGB")
    return Image.fromarray(pixels)


@pytest.fixture(scope="session")
def skelex_released_error_map(skelex_walkthrough_example):
    """Recompute the released pipeline's raw error map for the same masks.

    Mirrors ``mae_forward`` and ``compute_hotspot_map`` in the released
    ``unsupervised_heatmap.py``: denormalize with the processor statistics,
    clamp to [0, 1], and average the squared channel error over masked passes.
    The CPU generator repeats the tool's seeded noise sequence, so both runs
    see identical masks and the comparison isolates the arithmetic.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch
    from transformers import ViTImageProcessor, ViTMAEForPreTraining

    image, _, _ = skelex_walkthrough_example
    model, processor = get_hf_model_and_processor(
        "skhoha/SKELEX", HFModelType.IMAGE_PRETRAINING, use_fast=False
    )
    model = cast(ViTMAEForPreTraining, model)
    processor = cast(ViTImageProcessor, processor)
    source = _official_preprocessed_image(image)
    pixels = processor(images=source, return_tensors="pt")["pixel_values"].to(DEVICE)
    patch_size = model.config.patch_size
    patch_count = (pixels.shape[-2] // patch_size) * (pixels.shape[-1] // patch_size)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    std = torch.tensor(processor.image_std, device=DEVICE)[None, :, None, None]
    mean = torch.tensor(processor.image_mean, device=DEVICE)[None, :, None, None]
    source_pixels = (pixels * std + mean).clamp(0, 1)
    error_sum = torch.zeros_like(pixels[0, 0])
    hits = torch.zeros_like(error_sum)
    with torch.inference_mode():
        for _ in range(PASSES):
            noise = torch.rand((1, patch_count), generator=generator).to(DEVICE)
            output = model(pixel_values=pixels, noise=noise)
            reconstruction = model.unpatchify(output.logits)
            restored = (reconstruction * std + mean).clamp(0, 1)
            mask = model.unpatchify(
                output.mask[..., None].expand(-1, -1, patch_size**2 * 3)
            )[0, 0]
            error_sum += (restored - source_pixels).square().mean(dim=1)[0] * mask
            hits += mask
    observed = hits > 0
    error = error_sum / hits.clamp_min(1)
    return {
        "error": error.cpu().numpy(),
        "observed": observed.cpu().numpy(),
        "maximum": float(error.max().item()),
        "observed_mean": float(error[observed].mean().item()),
        "preprocessed": np.asarray(source),
        "config": model.config,
    }


def _resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a released pixel mask onto the processor canvas without blurring."""
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


@skip_on_import_error
@save_returns(unpack=True)
def test_skelex_reproduces_released_error_map(
    skelex_walkthrough_example, skelex_released_error_map
):
    """Reproduce the released error map and the paper's localization claim."""
    image, ground_truth, row = skelex_walkthrough_example
    reference = skelex_released_error_map
    config = reference["config"]
    # These are the released checkpoint's settings, not tool defaults.
    assert config.norm_pix_loss is False
    assert config.mask_ratio == pytest.approx(0.75)
    assert config.image_size == 224
    assert config.patch_size == 16
    # The tool applies the same image-wise min/max conversion as the release.
    assert np.array_equal(
        np.asarray(_prepare_skelex_image(image)), reference["preprocessed"]
    )

    result = skelex_reconstruction_error(image, passes=PASSES, seed=SEED)
    heatmap = decode_as(result)
    assert heatmap.size == (224, 224)
    assert heatmap.mode == "RGBA"
    pixels = np.asarray(heatmap)
    values = pixels[..., 0]
    observed = pixels[..., 3] > 0
    # Ten masking passes leave essentially every pixel with an estimate.
    assert observed.mean() > 0.99
    # Per-image scaling maps the largest observed error to full white.
    assert values.max() == 255
    # The map is grayscale: all three color channels hold the same error level.
    assert np.array_equal(pixels[..., 1], values)
    assert np.array_equal(pixels[..., 2], values)
    # Alpha is the observation mask, not an error quantity.
    assert set(np.unique(pixels[..., 3])) <= {0, 255}
    # Metadata identifies the model, sampling settings, and actual coverage.
    assert heatmap.info["model_id"] == "skhoha/SKELEX"
    assert heatmap.info["mask_ratio"] == "0.75"
    assert heatmap.info["passes"] == str(PASSES)
    assert heatmap.info["seed"] == str(SEED)
    assert float(heatmap.info["observed_pixel_fraction"]) == pytest.approx(
        observed.mean()
    )

    # The tool's raw error statistics and quantized map must match the released
    # pipeline computed above with the same seeded masks.
    assert np.array_equal(observed, reference["observed"])
    assert float(heatmap.info["maximum_mean_squared_error"]) == pytest.approx(
        reference["maximum"], rel=1e-6
    )
    assert float(heatmap.info["observed_pixel_mean_squared_error"]) == pytest.approx(
        reference["observed_mean"], rel=1e-6
    )
    expected_gray = np.zeros_like(reference["error"], dtype=np.uint8)
    if reference["maximum"] > 0:
        expected_gray = np.clip(
            reference["error"] / reference["maximum"] * 255, 0, 255
        ).astype(np.uint8)
    # The tool truncates the scaled error to 8 bits; allow one level for
    # device-level floating point differences.
    assert np.abs(values.astype(np.int16) - expected_gray.astype(np.int16)).max() <= 1

    target = _resize_binary_mask(ground_truth, heatmap.size)
    assert target.any(), "resized BTXRD tumor mask is empty"
    # The 8-bit map resolves 1/255 of the image's maximum error; smaller errors
    # truncate to zero and form a large plateau, so the paper's top-15% cut
    # would select mostly arbitrary ties. The smallest positive level is the
    # finest resolvable threshold and selects about 11.5% of the canvas here,
    # within the paper's 15% area budget.
    prediction = observed & (values > 0)
    assert 0 < prediction.mean() < 0.15
    dice = mask_overlap(prediction, target).dice
    assert dice is not None
    # Single-image analogue of the paper's paired test: estimate the null from
    # 1000 area-matched random masks and require P < 0.01 (the paper reports
    # P < 0.0001 over its BTXRD cohort).
    rng = np.random.default_rng(SEED)
    candidates = np.flatnonzero(observed)
    baselines = []
    for _ in range(RANDOM_MASK_TRIALS):
        choice = rng.choice(candidates, size=int(prediction.sum()), replace=False)
        random_mask = np.zeros(values.size, dtype=bool)
        random_mask[choice] = True
        baseline = mask_overlap(random_mask.reshape(values.shape), target).dice
        assert baseline is not None
        baselines.append(baseline)
    p_value = (1 + sum(baseline >= dice for baseline in baselines)) / (
        RANDOM_MASK_TRIALS + 1
    )
    assert p_value < 0.01
    return {
        "source": image,
        "ground_truth": encode_as(Image.fromarray(target.astype(np.uint8) * 255)),
        "heatmap": result,
        "metadata": dict(heatmap.info),
        "localization": {
            "dataset": "BTXRD",
            "image_id": row["image_id"],
            "lesion_label": "osteosarcoma",
            "threshold": "smallest positive 8-bit error",
            "selected_pixel_fraction": float(prediction.mean()),
            "dice": dice,
            "random_mask_dice_mean": float(np.mean(baselines)),
            "random_mask_dice_max": float(np.max(baselines)),
            "random_mask_trials": RANDOM_MASK_TRIALS,
            "empirical_p_value": p_value,
        },
    }


@skip_on_import_error
@save_returns(unpack=True)
def test_skelex_pass_coverage_and_seed_reproducibility(skelex_walkthrough_example):
    """Check pass coverage, seed-selected masks, and global-RNG isolation."""
    import torch

    image, _, _ = skelex_walkthrough_example
    rng_state = torch.random.get_rng_state().clone()
    one = decode_as(skelex_reconstruction_error(image, passes=1, seed=42))
    # Seeding is internal: the tool must not consume torch's global RNG.
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    repeated = decode_as(skelex_reconstruction_error(image, passes=1, seed=42))
    other = decode_as(skelex_reconstruction_error(image, passes=1, seed=43))
    for heatmap, passes, seed in ((one, "1", "42"), (other, "1", "43")):
        assert heatmap.info["passes"] == passes
        assert heatmap.info["seed"] == seed
        assert heatmap.info["mask_ratio"] == "0.75"
        observed = np.asarray(heatmap)[..., 3] > 0
        # One pass keeps 75% of the patch grid; every patch has the same
        # pixel area, so this is also the observed pixel fraction.
        assert observed.mean() == pytest.approx(0.75, abs=2e-3)
        assert float(heatmap.info["observed_pixel_fraction"]) == pytest.approx(
            observed.mean()
        )
    one_pixels = np.asarray(one)
    assert np.array_equal(one_pixels, np.asarray(repeated))
    # A different seed masks a different set of patches, visible in alpha.
    assert not np.array_equal(one_pixels[..., 3], np.asarray(other)[..., 3])
    three = decode_as(skelex_reconstruction_error(image, passes=3, seed=42))
    assert three.info["passes"] == "3"
    assert three.info["seed"] == "42"
    # Each pass masks 25% of the patch grid; three independent passes leave
    # roughly 0.25**3 unobserved, or about 98.4% coverage.
    three_observed = np.asarray(three)[..., 3] > 0
    assert three_observed.mean() == pytest.approx(1 - 0.25**3, abs=2e-3)
    assert float(three.info["observed_pixel_fraction"]) == pytest.approx(
        three_observed.mean()
    )
    return {
        "one_pass_seed_42": encode_as(one),
        "one_pass_seed_43": encode_as(other),
        "three_pass_seed_42": encode_as(three),
    }


@skip_on_import_error
@save_returns(unpack=True)
def test_skelex_palette_matches_rgb_rendering(skelex_walkthrough_example):
    """Check a palette-encoded image and its RGB rendering produce the same map."""
    image, _, _ = skelex_walkthrough_example
    palette = decode_as(image, mode="RGB").quantize(colors=64)
    assert palette.mode == "P"
    palette_image = encode_as(palette)
    rgb_image = encode_as(palette.convert("RGB"))
    assert decode_as(palette_image).mode == "P"
    # Both encodings carry identical colors; only the storage mode differs.
    assert np.array_equal(
        np.asarray(decode_as(palette_image, mode="RGB")),
        np.asarray(decode_as(rgb_image)),
    )
    result = skelex_reconstruction_error(palette_image, passes=1, seed=42)
    assert result == skelex_reconstruction_error(rgb_image, passes=1, seed=42)
    return {"source": palette_image, "heatmap": result}
