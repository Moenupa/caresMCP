"""Check MONAI pathology tools against the released model-zoo bundles.

The pinned bundles carry these published validation scores:

* ``MONAI/pathology_tumor_detection`` 0.6.3: accuracy 0.90 and FROC 0.72 on
  Camelyon16 patches.
* ``MONAI/pathology_nuclei_classification`` 0.2.1: F1 0.852 on CoNSeP.
* ``MONAI/pathology_nuclei_segmentation_classification`` 0.2.7: binary Dice
  0.8291, PQ 0.4973, and F1d 0.7417 on CoNSeP.

MONAI does not redistribute either dataset: CoNSeP sits behind a Warwick
sign-in and Camelyon16 requires challenge registration, so the dataset-level
scores cannot be re-measured here. These tests pin the metrics stored in the
released bundle metadata, then check every tool against the bundle's own
preprocessing, checkpoint, and inference settings on the official H&E tile
shipped with CONCH (the only pathology image released by an official model
repository in this package).
"""

import json
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("monai")

from caresmcp.api.decorators import download_bytes, save_returns
from caresmcp.api.io import decode_as, encode_as, get_classification_scores
from caresmcp.api.models import get_monai_model
from caresmcp.api.models.monai import MONAI_MODEL_REVISIONS, MonaiModels
from caresmcp.constants import DEVICE
from caresmcp.pathology.monai import (
    HOVERNET_ID,
    NUCLEI_CLASSES,
    NUCLEI_CLASSIFIER_ID,
    TUMOR_MODEL_ID,
    TUMOR_PATCH_SIZE,
    monai_pathology_nuclei_classification,
    monai_pathology_nuclei_segmentation_classification,
    monai_pathology_tumor_detection,
)

pytestmark = pytest.mark.slow

# Published metrics stored in each released bundle's configs/metadata.json.
BUNDLE_CLAIMS: dict[MonaiModels, dict[str, Any]] = {
    TUMOR_MODEL_ID: {
        "revision": "0.6.3",
        "eval_metrics": {"accuracy": 0.9, "froc": 0.72},
    },
    NUCLEI_CLASSIFIER_ID: {
        "revision": "0.2.1",
        "eval_metrics": {"f1_score": 0.852},
    },
    HOVERNET_ID: {
        "revision": "0.2.7",
        "eval_metrics": {"Binary Dice": 0.8291},
    },
}
# HoVerNet's README additionally publishes PQ 0.4973 and F1d 0.7417.
HOVERNET_README_CLAIMS = ("PQ: 0.4973", "F1d: 0.7417")

# The released inference config processes 256x256 windows with 164x164 output
# centers and pads each window by (256 - 164) // 2 on every side.
HOVERNET_WINDOW = 256
HOVERNET_OUTPUT = 164
HOVERNET_PADDING = (HOVERNET_WINDOW - HOVERNET_OUTPUT) // 2


def _bundle_file_url(model_id: str, revision: str, path: str) -> str:
    """Return a revision-pinned download URL for one bundle file."""
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{path}"


@pytest.fixture(scope="module")
def region(pathology_image: bytes) -> Image.Image:
    """Crop a 512x512 region from the official H&E tile."""
    source = decode_as(pathology_image, mode="RGB")
    x, y = source.width // 2, source.height // 2
    return source.crop((x - 256, y - 256, x + 256, y + 256))


@pytest.mark.parametrize("model_id", tuple(BUNDLE_CLAIMS))
def test_released_bundles_carry_published_scores(model_id: MonaiModels):
    """Check the pinned releases still publish their claimed metrics."""
    claim = BUNDLE_CLAIMS[model_id]
    assert MONAI_MODEL_REVISIONS[model_id] == claim["revision"]
    metadata = json.loads(
        download_bytes(
            _bundle_file_url(model_id, claim["revision"], "configs/metadata.json")
        )
    )
    assert metadata["eval_metrics"] == claim["eval_metrics"]
    if model_id == HOVERNET_ID:
        readme = download_bytes(
            _bundle_file_url(model_id, claim["revision"], "docs/README.md")
        ).decode()
        for published in HOVERNET_README_CLAIMS:
            assert published in readme


def _reference_tumor_scores(pixels: np.ndarray) -> list[float]:
    """Score the tool's patch grid with the bundle preprocessing and model."""
    import torch
    from monai.transforms import ScaleIntensityRanged

    scale = ScaleIntensityRanged(
        keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True
    )
    model = get_monai_model(TUMOR_MODEL_ID)
    height, width = pixels.shape[:2]
    scores = []
    with torch.inference_mode():
        for top in range(0, height, TUMOR_PATCH_SIZE):
            for left in range(0, width, TUMOR_PATCH_SIZE):
                right = min(left + TUMOR_PATCH_SIZE, width)
                bottom = min(top + TUMOR_PATCH_SIZE, height)
                # The tool pads partial edge patches with white, like the
                # bundle's patch extraction at the slide boundary.
                patch = np.full(
                    (TUMOR_PATCH_SIZE, TUMOR_PATCH_SIZE, 3), 255, dtype=np.float32
                )
                patch[: bottom - top, : right - left] = pixels[top:bottom, left:right]
                image = scale(
                    {"image": torch.from_numpy(patch).permute(2, 0, 1).float()}
                )["image"]
                tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
                logits = model(tensor.unsqueeze(0).to(DEVICE))
                scores.append(float(torch.sigmoid(logits).item()))
    return scores


@save_returns("monai_pathology")
def test_tumor_detection_matches_released_bundle(region: Image.Image):
    """Reproduce the bundle's patch scores and boxes on the official tile."""
    result = monai_pathology_tumor_detection(encode_as(region), batch_size=4)
    pixels = np.asarray(region)
    height, width = pixels.shape[:2]
    expected_boxes = [
        [x, y, min(x + TUMOR_PATCH_SIZE, width), min(y + TUMOR_PATCH_SIZE, height)]
        for y in range(0, height, TUMOR_PATCH_SIZE)
        for x in range(0, width, TUMOR_PATCH_SIZE)
    ]
    assert [patch["box"] for patch in result["patches"]] == expected_boxes
    scores = [patch["tumor_score"] for patch in result["patches"]]
    # The tool scales patches with ``pixels / 127.5 - 1`` and the bundle with
    # MONAI's ScaleIntensityRanged; float32 rounding can accumulate through
    # the network, so allow a small absolute difference.
    assert scores == pytest.approx(_reference_tumor_scores(pixels), abs=2e-3)
    assert result["max_tumor_score"] == max(scores)
    return result


@pytest.fixture(scope="module")
def nuclei(region: Image.Image) -> dict[str, Any]:
    """Run the released HoVerNet bundle once for both nuclei tests."""
    return monai_pathology_nuclei_segmentation_classification(
        encode_as(region), batch_size=4
    )


@pytest.fixture(scope="module")
def reference_nuclei_maps(region: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Run the released HoVerNet pipeline with the bundle preprocessing."""
    import torch
    from monai.apps.pathology.inferers import SlidingWindowHoVerNetInferer
    from monai.apps.pathology.transforms import (
        HoVerNetInstanceMapPostProcessing,
        HoVerNetNuclearTypePostProcessing,
    )
    from monai.transforms import ScaleIntensityRanged

    pixels = np.asarray(region, dtype=np.float32)
    # The bundle loads PNGs with MONAI's PILReader, which returns channels
    # first, then scales intensities to [0, 1].
    image = ScaleIntensityRanged(
        keys="image", a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
    )({"image": pixels.transpose(2, 1, 0).copy()})["image"]
    inferer = SlidingWindowHoVerNetInferer(
        roi_size=HOVERNET_WINDOW,
        sw_batch_size=4,
        overlap=1.0 - float(HOVERNET_OUTPUT) / float(HOVERNET_WINDOW),
        padding_mode="constant",
        cval=0,
        extra_input_padding=(HOVERNET_PADDING,) * 4,  # ty: ignore[invalid-argument-type]
        sw_device=DEVICE,
        device="cpu",
    )
    with torch.inference_mode():
        predictions = cast(
            dict[str, torch.Tensor],
            inferer(
                torch.from_numpy(np.array(image)).unsqueeze(0),
                get_monai_model(HOVERNET_ID),
            ),
        )
    info, instance_map = HoVerNetInstanceMapPostProcessing(
        sobel_kernel_size=21, marker_threshold=0.4, marker_radius=2
    )(predictions["nucleus_prediction"][0], predictions["horizontal_vertical"][0])
    _, type_map = HoVerNetNuclearTypePostProcessing()(
        predictions["type_prediction"][0], info, instance_map
    )

    def values(tensor) -> np.ndarray:
        array = tensor.as_tensor() if hasattr(tensor, "as_tensor") else tensor
        return np.asarray(array.detach().cpu()).squeeze(0)

    instances = values(instance_map).T.astype(np.int32)
    types = values(type_map).T.astype(np.uint8)
    return instances, types


@save_returns("monai_pathology")
def test_nuclei_segmentation_matches_released_bundle(
    nuclei: dict[str, Any], reference_nuclei_maps: tuple[np.ndarray, np.ndarray]
):
    """Reproduce the bundle's instance and type maps exactly."""
    instances = np.asarray(nuclei["instance_map"])
    types = np.asarray(nuclei["type_map"])
    expected_instances, expected_types = reference_nuclei_maps
    np.testing.assert_array_equal(instances, expected_instances)
    np.testing.assert_array_equal(types, expected_types)
    assert set(np.unique(types)) <= set(nuclei["labels"])
    assert nuclei["instance_count"] == np.unique(instances[instances > 0]).size > 0
    return nuclei


@save_returns("monai_pathology")
def test_nuclei_classification_matches_released_bundle(
    region: Image.Image, nuclei: dict[str, Any]
):
    """Reproduce the bundle's guided nucleus classification exactly."""
    import torch
    from monai.apps.nuclick.transforms import AddLabelAsGuidanced
    from monai.transforms import ScaleIntensityRanged

    instances = np.asarray(nuclei["instance_map"])
    positions = np.argwhere(instances > 0)
    assert len(positions), "Expected nuclei in the official tile"
    y, x = positions[len(positions) // 2]
    left = min(max(int(x) - 64, 0), region.width - 128)
    top = min(max(int(y) - 64, 0), region.height - 128)
    crop = region.crop((left, top, left + 128, top + 128))
    mask = instances[top : top + 128, left : left + 128] == instances[y, x]

    result = monai_pathology_nuclei_classification(
        encode_as(crop), encode_as(Image.fromarray(mask.astype(np.uint8) * 255))
    )

    # The bundle's inference config loads the patch channels-first, scales
    # intensities to [-1, 1], then appends the label mask as a guidance
    # channel through AddLabelAsGuidanced.
    image = ScaleIntensityRanged(
        keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True
    )({"image": np.asarray(crop, dtype=np.float32).transpose(2, 1, 0)})["image"]
    guided = AddLabelAsGuidanced(keys="image", source="label")(
        {"image": np.array(image), "label": np.ascontiguousarray(mask.T)}
    )
    with torch.inference_mode():
        logits = get_monai_model(NUCLEI_CLASSIFIER_ID)(
            torch.as_tensor(np.array(guided["image"])).unsqueeze(0).to(DEVICE)
        )
    expected = get_classification_scores(
        logits, dict(enumerate(NUCLEI_CLASSES)), "softmax"
    )
    assert result["class_scores"] == pytest.approx(expected, abs=1e-4)
    assert result["predicted_class"] == max(expected, key=expected.__getitem__)
    return result
