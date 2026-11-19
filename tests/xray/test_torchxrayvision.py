"""Exercise TorchXRayVision tools on real radiographs and the released demo.

The target Hernia score is checked against recorded values from a deterministic
CPU run, and the anatomy model is checked against the official ChestX-Det demo
output.
"""

import hashlib
from dataclasses import asdict
from io import BytesIO
from typing import Literal

import numpy as np
import pydicom
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import decode_as, encode_as
from caresmcp.api.metrics import mask_overlap
from caresmcp.api.models import get_torchxrayvision_model
from caresmcp.xray.torchxrayvision import (
    ANATOMICAL_TARGETS,
    ChestXrayWeights,
    torchxrayvision_classify,
    torchxrayvision_localize,
    torchxrayvision_segment_anatomy,
)

pytestmark = pytest.mark.slow

# Labels exposed by each checkpoint. Values that a checkpoint exposes but
# cannot score are listed in EXPECTED_UNAVAILABLE.
NIH_LABELS = frozenset(
    {
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
    }
)
ALL_LABELS = NIH_LABELS | {
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
}
EXPECTED_LABELS: dict[ChestXrayWeights, frozenset[str]] = {
    "densenet121-res224-all": ALL_LABELS,
    "densenet121-res224-nih": NIH_LABELS,
    "densenet121-res224-pc": NIH_LABELS | {"Fracture"},
    "resnet50-res512-all": ALL_LABELS,
}
EXPECTED_UNAVAILABLE: dict[ChestXrayWeights, frozenset[str]] = {
    "resnet50-res512-all": frozenset({"Lung Lesion", "Enlarged Cardiomediastinum"}),
}

# Recorded Hernia scores from a deterministic CPU run on the NIH radiograph.
# These are operating-point-normalized outputs, not probabilities: each
# checkpoint maps its own per-label threshold to 0.5, so values are not
# comparable across checkpoints. The nih-only checkpoint is weakest on Hernia
# (raw sigmoid ~0.07, mapped to 0.5371 by its 0.00044 threshold; NIH has 227
# positives in 112,120). The four selected checkpoints cover both released
# architectures and each distinct label schema while bounding slow downloads.
EXPECTED_HERNIA_SCORES: dict[ChestXrayWeights, float] = {
    "densenet121-res224-all": 0.9981,
    "densenet121-res224-nih": 0.5371,
    "densenet121-res224-pc": 0.9298,
    "resnet50-res512-all": 0.9637,
}

# Official ChestX-Det release, fetched from a revision-pinned jsDelivr mirror:
# the released PSPNet's own demo input and the Left Lung PNG written by its
# pre-trained PSPNet script. The TorchXRayVision checkpoint shares the released
# weight filename; residual Dice loss comes from the official OpenCV cubic
# resize versus TorchXRayVision's skimage resize, so the test requires high
# agreement instead of pixel equality.
CHESTX_DET_REVISION = "a33fa5ce55f58b0f2b7b7977c9ed5c313283303a"
CHESTX_DET_DEMO_ROOT = (
    "https://cdn.jsdelivr.net/gh/Deepwise-AILab/ChestX-Det-Dataset@"
    f"{CHESTX_DET_REVISION}/pre-trained_PSPNet/"
)
CHESTX_DET_DEMO_SHA256: dict[str, str] = {
    "demo.png": "04b94409c88760a347bcd614f571922caf616662257f49ac71835039232f4743",
    "demo_left_lung.png": (
        "64635475cf40974a9dc52f23b1147293e85c68daf6d45d417383fed6eacb4f26"
    ),
}
MINIMUM_DEMO_DICE = 0.99


@pytest.fixture(scope="session")
def chestx_det_demo() -> tuple[bytes, bytes]:
    """Download the official ChestX-Det PSPNet demo image and Left Lung mask."""
    contents = {
        name: _read_bytes(CHESTX_DET_DEMO_ROOT + name)
        for name in CHESTX_DET_DEMO_SHA256
    }
    for name, content in contents.items():
        digest = hashlib.sha256(content).hexdigest()
        assert digest == CHESTX_DET_DEMO_SHA256[name], (
            f"{name}: official artifact bytes changed"
        )
    return contents["demo.png"], contents["demo_left_lung.png"]


@pytest.mark.parametrize("weights", tuple(EXPECTED_HERNIA_SCORES.keys()))
@skip_on_import_error
@save_returns()
def test_torchxrayvision_classify(hernia_image: bytes, weights: ChestXrayWeights):
    """Check the checkpoint, label set, unavailable labels, and Hernia score."""
    result = torchxrayvision_classify(hernia_image, weights=weights)

    assert result["model_id"] == weights
    assert result["score_mode"] == "sigmoid"
    scores = result["scores"]
    assert set(scores) == EXPECTED_LABELS[weights]
    unavailable = {name for name, score in scores.items() if score is None}
    assert unavailable == EXPECTED_UNAVAILABLE.get(weights, frozenset())
    available = {name: score for name, score in scores.items() if score is not None}
    assert available
    assert all(
        isinstance(score, float) and 0 <= score <= 1 for score in available.values()
    )

    # The study contains a hernia; every selected checkpoint must detect it and
    # reproduce the recorded score.
    hernia_score = scores["Hernia"]
    assert hernia_score is not None
    assert hernia_score == pytest.approx(EXPECTED_HERNIA_SCORES[weights], rel=1e-2)
    assert hernia_score > 0.5
    return result


@skip_on_import_error
def test_torchxrayvision_segment_anatomy_target_order():
    """Check mask IDs follow the shared anatomy model's target order."""
    model = get_torchxrayvision_model("anatomy")
    assert tuple(model.targets) == ANATOMICAL_TARGETS


@skip_on_import_error
@save_returns(unpack=True)
def test_torchxrayvision_segment_anatomy(
    chestx_det_demo: tuple[bytes, bytes],
):
    """Reproduce the official demo and check the mask output contracts.

    ChestX-Det publishes ``demo.png`` together with the Left Lung PNG produced
    by its own pre-trained PSPNet script. Require Dice >= 0.99 so the check
    tolerates the different resize engines while still pinning the released
    checkpoint and the tool's preprocessing. The same official radiograph also
    exercises the region, legend, and label-mask output contracts, including the
    documented agreement between a region mask and the combined mask.
    """
    image, published_png = chestx_det_demo
    published = np.asarray(decode_as(published_png, mode="L")) > 0
    assert published.any(), "Expected the published demo to contain a Left Lung"

    prediction_png = torchxrayvision_segment_anatomy(image, region="Left Lung")
    prediction_image = decode_as(prediction_png)
    assert prediction_image.format == "PNG"
    assert prediction_image.mode == "L"
    assert prediction_image.size == (512, 512)
    prediction_values = np.asarray(prediction_image)
    assert set(np.unique(prediction_values)) <= {0, 255}
    prediction = prediction_values > 0
    assert prediction.shape == published.shape
    overlap = mask_overlap(prediction, published)
    assert overlap.dice is not None, "Expected both demo masks to be nonempty"
    assert overlap.dice >= MINIMUM_DEMO_DICE, (
        f"Measured Dice {overlap.dice:.4f} does not reproduce the official demo"
    )

    artifacts = {
        "source": image,
        "prediction": prediction_png,
        "published_mask": published_png,
    }
    core_ids = {
        ANATOMICAL_TARGETS.index(name) + 1
        for name in (
            "Left Lung",
            "Right Lung",
            "Heart",
            "Facies Diaphragmatica",
            "Spine",
        )
    }
    masks = {}
    for output in ("legend", "label_mask"):
        result = torchxrayvision_segment_anatomy(image, output=output)
        mask = decode_as(result)
        assert mask.format == "PNG"
        assert mask.mode == "P"
        assert mask.height == 512
        if output == "label_mask":
            assert mask.size == (512, 512)
        else:
            assert mask.width > 512
        assert mask.info["transparency"] == 0
        palette = mask.getpalette()
        assert palette is not None and len(palette) == 768
        assert palette[:3] == [0, 0, 0], "Background must be black"
        labels = np.asarray(mask)[:, :512]
        assert set(np.unique(labels)) <= set(range(len(ANATOMICAL_TARGETS) + 1))
        assert core_ids <= set(np.unique(labels)), (
            "Expected core anatomy in the official demo radiograph"
        )
        masks[output] = mask
        artifacts[f"demo_{output}"] = result

    combined_left_lung = (
        np.asarray(masks["label_mask"])[:, :512]
        == ANATOMICAL_TARGETS.index("Left Lung") + 1
    )
    assert combined_left_lung.any(), "Expected the left lung in the official demo"
    # Region masks threshold one structure independently, so they retain every
    # pixel that the combined highest-score mask assigned to that structure.
    assert np.all(prediction[combined_left_lung])
    agreement = mask_overlap(prediction, combined_left_lung).dice
    assert agreement is not None and agreement > 0.8
    return {
        **artifacts,
        "metrics": asdict(overlap)
        | {"minimum_dice": MINIMUM_DEMO_DICE, "region_combined_dice": agreement},
    }


@pytest.mark.parametrize(
    ("weights", "size"), [("densenet121-res224-all", 224), ("resnet50-res512-all", 512)]
)
@pytest.mark.parametrize("output", ["overlay", "saliency"])
@skip_on_import_error
@save_returns()
def test_torchxrayvision_localize(
    hernia_image: bytes,
    weights: ChestXrayWeights,
    size: int,
    output: Literal["overlay", "saliency"],
):
    """Check Grad-CAM PNGs from both supported classifier architectures."""
    result = torchxrayvision_localize(
        hernia_image, label="Hernia", weights=weights, output=output
    )
    image = decode_as(result)
    assert image.format == "PNG"
    assert image.size == (size, size)
    assert image.mode == ("RGB" if output == "overlay" else "L")
    pixels = np.asarray(image)
    assert pixels.dtype == np.uint8
    assert pixels.any()
    if output == "overlay":
        # A plain grayscale radiograph would have equal color channels.
        assert not np.array_equal(pixels[..., 0], pixels[..., 1])
        assert not np.array_equal(pixels[..., 1], pixels[..., 2])
    else:
        # compute_gradcam scales the positive map to the full uint8 range.
        assert pixels.min() == 0
        assert pixels.max() == 255
        if weights == "densenet121-res224-all":
            # Saliency must depend on the requested finding, not just the input.
            effusion = np.asarray(
                decode_as(
                    torchxrayvision_localize(
                        hernia_image, label="Effusion", weights=weights, output=output
                    )
                ),
                dtype=np.int16,
            )
            assert np.abs(pixels.astype(np.int16) - effusion).mean() > 10
    return result


@skip_on_import_error
def test_torchxrayvision_localize_unavailable_label(hernia_image: bytes):
    """Check labels without an operating threshold fail instead of rendering a blank map."""
    with pytest.raises(ValueError, match="no operating threshold"):
        torchxrayvision_localize(
            hernia_image, label="Lung Lesion", weights="resnet50-res512-all"
        )


@skip_on_import_error
@save_returns()
def test_torchxrayvision_classify_dicom(chest_dicom: bytes):
    """Check DICOM input scores identically to its stored pixels supplied as PNG."""
    dataset = pydicom.dcmread(BytesIO(chest_dicom))
    pixels = np.asarray(dataset.pixel_array, dtype=np.uint8)
    raster = encode_as(Image.fromarray(pixels))
    dicom_result = torchxrayvision_classify(chest_dicom)
    raster_result = torchxrayvision_classify(raster)
    assert dicom_result["model_id"] == "densenet121-res224-all"
    assert dicom_result == raster_result
    assert any(score is not None for score in dicom_result["scores"].values())
    return dicom_result
