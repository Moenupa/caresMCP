"""Measure FleXray on official released image/mask pairs.

Every case is a test pair from
https://huggingface.co/datasets/VictorButoi/flexray-data, read from a pinned
snapshot. The reproduction metric follows the ``FleXray (single)`` row of
Table 6 in https://arxiv.org/abs/2609.26756: per-image mean label Dice at the
default 0.5 threshold, reference structures below 0.1% image area dropped, and
HipRay collapsed to an exclusive femurs-over-hips overlay (Methods C.1-C.2).

Running that metric over every published test image reproduced Table 6 within
0.05% relative: ElbowLat 0.85444 vs 0.854, HipRay 0.93727 vs 0.937, and
LowerLimbs 0.83091 vs 0.831. These tests evaluate the first and middle
published test sample of each dataset, keeping the suite under ten images, and
pin their per-image Dice within 1% relative. The remaining tests cover the
released model contract and mask semantics on the same official pairs.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any, get_args
from urllib.parse import quote

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import decode_as, encode_as
from caresmcp.api.metrics import mask_overlap
from caresmcp.xray.flexray import AnatomicalRegion, flexray_segment_anatomy

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

# Official FleXray release: real image/mask pairs for the held-out datasets
# whose published Dice these tests reproduce.
FLEXRAY_DATASET_ID = "VictorButoi/flexray-data"
FLEXRAY_DATA_REVISION = "1aeffa9e4d9a1b72416bb5e896bf07321b9f3fc4"
FLEXRAY_DATA_ROOT = (
    "https://huggingface.co/datasets/"
    f"{FLEXRAY_DATASET_ID}/resolve/{FLEXRAY_DATA_REVISION}"
)
FLEXRAY_DATASETS: dict[str, dict[str, Any]] = {
    "ElbowLat": {
        "regions": ("humeri", "radii", "ulnae"),
        "test_count": 91,
        "overlapping": True,
    },
    "HipRay": {
        "regions": ("femurs", "hips"),
        "test_count": 21,
        "overlapping": False,
    },
    "LowerLimbs": {
        "regions": ("femurs", "fibulae", "tibiae"),
        "test_count": 8,
        "overlapping": True,
    },
}


@dataclass(frozen=True)
class FlexrayReleasedDataset:
    """One official FleXray test split sampled down to two published pairs.

    Attributes:
        name: Released dataset folder name.
        revision: Hugging Face snapshot that owns the dataset manifest.
        samples: First and middle published test samples.
        regions: Dataset-native labels scored by the FleXray protocol.
        overlapping: Whether the released evaluation masks may overlap.
        read_sample: Return one sample's image and label bytes.
        reference_masks: Load the fixed regions as boolean arrays.
    """

    name: str
    revision: str
    samples: tuple[dict[str, Any], ...]
    regions: tuple[str, ...]
    overlapping: bool
    read_sample: Callable[[dict[str, Any]], tuple[bytes, bytes]]
    reference_masks: Callable[[bytes], dict[str, np.ndarray]]


@pytest.fixture(scope="session", params=tuple(FLEXRAY_DATASETS))
def flexray_released_dataset(request):
    """Supply two official FleXray test pairs from a pinned data snapshot."""
    pytest.importorskip("fxr")
    yaml = pytest.importorskip("yaml")

    name = request.param
    config = FLEXRAY_DATASETS[name]
    regions: tuple[str, ...] = config["regions"]
    manifest = yaml.safe_load(_read_bytes(f"{FLEXRAY_DATA_ROOT}/{name}/dataset.yml"))
    assert manifest["dataset_name"] == name, "Released manifest identity changed"
    test_samples = [
        sample for sample in manifest["samples"] if sample["split"] == "test"
    ]
    assert len(test_samples) == config["test_count"], (
        f"{name}: published test split size changed"
    )
    test_subjects = {sample["subject_id"] for sample in test_samples}
    other_subjects = {
        sample["subject_id"]
        for sample in manifest["samples"]
        if sample["split"] in {"train", "val"}
    }
    assert not test_subjects & other_subjects, (
        f"{name}: test subject IDs overlap another published partition"
    )
    label_names = manifest.get("label_names") or list(
        manifest["stored_labels"].values()
    )
    missing = set(regions) - set(label_names)
    assert not missing, f"{name}: manifest misses regions {sorted(missing)}"
    # Fixed subset: the first and middle published test samples keep the
    # evaluation under ten images while spanning two separate subjects.
    samples = (test_samples[0], test_samples[len(test_samples) // 2])

    def read_sample(sample: dict[str, Any]) -> tuple[bytes, bytes]:
        """Read one image/label pair from the revision-pinned snapshot."""
        image = _read_bytes(f"{FLEXRAY_DATA_ROOT}/{name}/{quote(sample['image'])}")
        label = _read_bytes(f"{FLEXRAY_DATA_ROOT}/{name}/{quote(sample['label'])}")
        return image, label

    def reference_masks(label: bytes) -> dict[str, np.ndarray]:
        """Load the dataset's scored labels as boolean reference masks."""
        if "label_names" in manifest:
            channels = np.load(BytesIO(label), allow_pickle=False)
            names = manifest["label_names"]
            return {region: channels[names.index(region)] > 0 for region in regions}
        ids = {
            label_name: int(index)
            for index, label_name in manifest["stored_labels"].items()
        }
        labels = np.asarray(decode_as(label))
        return {region: labels == ids[region] for region in regions}

    return FlexrayReleasedDataset(
        name=name,
        revision=FLEXRAY_DATA_REVISION,
        samples=samples,
        regions=regions,
        overlapping=bool(config["overlapping"]),
        read_sample=read_sample,
        reference_masks=reference_masks,
    )


@pytest.fixture(scope="session")
def flexray_cached_model():
    """Load the flagship segmenter once and reuse it across region calls."""
    pytest.importorskip("fxr")
    from caresmcp.xray import flexray

    monkeypatch = pytest.MonkeyPatch()
    model = flexray.get_flexray_model()
    monkeypatch.setattr(flexray, "get_flexray_model", lambda: model)
    yield model
    monkeypatch.undo()


# FleXray (single) mean label Dice from Table 6 (arXiv:2609.26756v2), kept as
# provenance for the recorded subset values.
PUBLISHED_DICE: dict[str, float] = {
    "ElbowLat": 0.854,
    "HipRay": 0.937,
    "LowerLimbs": 0.831,
}
# Per-region Dice recorded from the full-test run that reproduced Table 6.
RECORDED_SAMPLE_DICE: dict[str, dict[str, float]] = {
    "test_115_jpg.rf.79XAF49uapsmI58ilCFw": {
        "humeri": 0.8859,
        "radii": 0.7082,
        "ulnae": 0.6674,
    },
    "test_253_jpg.rf.EPmFV2Mfb3mKdKhpPKsz": {
        "humeri": 0.9388,
        "radii": 0.8872,
        "ulnae": 0.8735,
    },
    "image_006": {"femurs": 0.9608, "hips": 0.9243},
    "image_066": {"femurs": 0.9656, "hips": 0.9373},
    "test_Knee 1_JPG.rf.AAY7S1HbBVKCNLqXgfOR": {
        "femurs": 0.9451,
        "fibulae": 0.9017,
        "tibiae": 0.9355,
    },
    "test_knee13_jpg.rf.ZIxrjMlPv6sAxPPpY49r": {
        "femurs": 0.9431,
        "fibulae": 0.8738,
        "tibiae": 0.9200,
    },
}
# HipRay masks partition the image; predictions and references are collapsed
# with femurs taking priority over hips before scoring.
EXCLUSIVE_PRIORITY: dict[str, tuple[str, ...]] = {"HipRay": ("femurs", "hips")}
MINIMUM_REFERENCE_FRACTION = 0.001


def _exclusive_overlay(
    masks: dict[str, np.ndarray], priority: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Assign each pixel to the highest-priority structure present."""
    overlay = np.zeros(next(iter(masks.values())).shape, dtype=np.uint8)
    for rank, region in enumerate(reversed(priority)):
        overlay[masks[region]] = len(priority) - rank
    return {region: overlay == index for index, region in enumerate(priority, start=1)}


@skip_on_import_error
def test_flexray_released_contract(flexray_cached_model):
    """Check the released bundle label order and preprocessing contract."""
    model = flexray_cached_model
    assert model.label_names == ("background", *get_args(AnatomicalRegion))
    assert model.subfolders == ("members/flux0375",)
    assert model.probability_mode == "multilabel"
    assert model.preprocessing.image_size == (256, 256)
    assert model.preprocessing.color_mode == "grayscale"
    assert model.preprocessing.pad_to_square is True
    assert model.preprocessing.normalization_scheme == "percentile_minmax"
    assert model.preprocessing.percentiles == (0.5, 99.5)


@skip_on_import_error
@save_returns("flexray", unpack=True)
def test_flexray_segment_anatomy(flexray_released_dataset, flexray_cached_model):
    """Check output geometry and agreement with each official reference mask."""
    case = flexray_released_dataset
    sample = case.samples[0]
    image, label = case.read_sample(sample)
    source = decode_as(image)
    reference = case.reference_masks(label)
    area = source.width * source.height
    artifacts = {"source": image, "sample_id": sample["sample_id"]}
    scored = False
    for region in case.regions:
        result = flexray_segment_anatomy(image, region)
        mask = decode_as(result)
        assert mask.size == source.size
        assert mask.mode == "L"
        pixels = np.asarray(mask)
        assert set(np.unique(pixels)) <= {0, 255}
        predicted = pixels > 0
        target = reference[region]
        if target.sum() >= MINIMUM_REFERENCE_FRACTION * area:
            assert predicted.any(), f"{case.name}/{region}: empty prediction"
            dice = mask_overlap(predicted, target).dice
            assert dice is not None and dice > 0.5
            scored = True
        artifacts[f"{region}_prediction"] = result
        artifacts[f"{region}_reference"] = encode_as(
            Image.fromarray(target.astype(np.uint8) * 255)
        )
    assert scored, f"{case.name}: no official structure reached the scoring area"
    return artifacts


@skip_on_import_error
@save_returns("flexray", unpack=True)
def test_flexray_probability_threshold(flexray_released_dataset, flexray_cached_model):
    """Check thresholds filter one official structure monotonically."""
    case = flexray_released_dataset
    image, _ = case.read_sample(case.samples[0])
    region = case.regions[0]
    results, masks = {}, {}
    for threshold in (0.0, 0.1, 0.5, 0.9):
        result = flexray_segment_anatomy(image, region, probability_threshold=threshold)
        pixels = np.asarray(decode_as(result))
        assert set(np.unique(pixels)) <= {0, 255}
        masks[threshold] = pixels > 0
        results[f"{region}_threshold_{threshold}"] = result
    # Each pixel is thresholded independently, so stricter thresholds retain a
    # subset of the pixels, and zero keeps every sigmoid output.
    assert masks[0.0].all()
    assert np.all(masks[0.1] <= masks[0.0])
    assert np.all(masks[0.5] <= masks[0.1])
    assert np.all(masks[0.9] <= masks[0.5])
    assert np.any(masks[0.1] & ~masks[0.5])
    assert np.any(masks[0.5] & ~masks[0.9])
    return results


@skip_on_import_error
@save_returns("flexray", unpack=True)
def test_flexray_masks_threshold_independently(
    flexray_released_dataset, flexray_cached_model
):
    """Check separate passes preserve projection overlap in official anatomy."""
    case = flexray_released_dataset
    image, label = case.read_sample(case.samples[0])
    reference = case.reference_masks(label)
    prediction = {
        region: np.asarray(decode_as(flexray_segment_anatomy(image, region))) > 0
        for region in case.regions
    }
    pairs = list(itertools.combinations(case.regions, 2))
    assert pairs, f"{case.name}: need at least two regions"
    for first, second in pairs:
        assert not np.array_equal(prediction[first], prediction[second])
    overlapping = [
        (first, second)
        for first, second in pairs
        if (reference[first] & reference[second]).any()
    ]
    if not case.overlapping:
        assert not overlapping, f"{case.name}: exclusive reference masks overlap"
    if overlapping:
        assert any(
            (prediction[first] & prediction[second]).any()
            for first, second in overlapping
        ), f"{case.name}: independent passes dropped a reference overlap"
    return {
        region: encode_as(Image.fromarray(mask.astype(np.uint8) * 255))
        for region, mask in prediction.items()
    }


@skip_on_import_error
@save_returns("flexray", unpack=True)
def test_flexray_released_annotation_dice(
    flexray_released_dataset, flexray_cached_model
):
    """Check official test pairs reproduce recorded per-image Dice within 1%."""
    case = flexray_released_dataset
    priority = EXCLUSIVE_PRIORITY.get(case.name)
    evaluated = []
    artifacts = {}
    for index, sample in enumerate(case.samples):
        expected = RECORDED_SAMPLE_DICE[sample["sample_id"]]
        image, label = case.read_sample(sample)
        reference = case.reference_masks(label)
        if priority is not None:
            reference = _exclusive_overlay(reference, priority)
        prediction = {
            region: np.asarray(decode_as(flexray_segment_anatomy(image, region))) > 0
            for region in case.regions
        }
        if priority is not None:
            prediction = _exclusive_overlay(prediction, priority)
        area = next(iter(reference.values())).size
        measured = {}
        for region in case.regions:
            assert prediction[region].shape == reference[region].shape, (
                f"{case.name}/{region}: prediction and reference grids differ"
            )
            if reference[region].sum() < MINIMUM_REFERENCE_FRACTION * area:
                assert region not in expected, (
                    f"{case.name}/{region}: scored region dropped below 0.1%"
                )
                continue
            overlap = mask_overlap(prediction[region], reference[region])
            assert overlap.dice is not None, (
                f"{case.name}/{region}: no scorable reference"
            )
            measured[region] = overlap.dice
            assert overlap.dice == pytest.approx(expected[region], rel=1e-2), (
                f"{case.name}/{sample['sample_id']}/{region}: measured Dice "
                f"{overlap.dice:.4f} does not reproduce {expected[region]:.4f}"
            )
        assert set(measured) == set(expected), (
            f"{case.name}/{sample['sample_id']}: scored regions changed"
        )
        image_dice = float(np.mean(list(measured.values())))
        assert image_dice == pytest.approx(np.mean(list(expected.values())), rel=1e-2)
        evaluated.append(
            {
                "sample_id": sample["sample_id"],
                "expected_dice": expected,
                "measured_dice": measured,
                "per_image_mean_dice": image_dice,
            }
        )
        if index == 0:
            artifacts = {"source": image, "sample_id": sample["sample_id"]}
            for region in case.regions:
                artifacts[f"{region}_prediction"] = encode_as(
                    Image.fromarray(prediction[region].astype(np.uint8) * 255)
                )
                artifacts[f"{region}_reference"] = encode_as(
                    Image.fromarray(reference[region].astype(np.uint8) * 255)
                )

    return {
        **artifacts,
        "metrics": {
            "source": (
                "https://huggingface.co/datasets/VictorButoi/flexray-data/tree/"
                f"{case.revision}/{case.name}"
            ),
            "source_data_revision": case.revision,
            "dataset": case.name,
            "split": "test",
            "published_source": "https://arxiv.org/abs/2609.26756",
            "published_table": "Table 6, FleXray (single)",
            "published_full_split_dice": PUBLISHED_DICE[case.name],
            "samples": evaluated,
            "metric": (
                "per-structure Dice averaged per image; reference structures "
                "below 0.1% image area dropped"
            ),
            "postprocessing": (
                "exclusive femurs-over-hips collapse" if priority else "none"
            ),
            "subset_note": (
                "First and middle published test samples; the full test split "
                "was verified to reproduce the published Dice within 0.05%"
            ),
        },
    }
