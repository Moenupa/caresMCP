"""Evaluate non-chest anatomy against released real-image annotations.

Sources, licenses, training/evaluation roles, and source-specific subject IDs:
https://huggingface.co/datasets/VictorButoi/flexray-data
https://huggingface.co/VictorButoi/flexray
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.api.io import encode_as
from caresmcp.xray.flexray import AnatomicalRegion, flexray_segment_anatomy

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

DATASET_REGIONS: dict[str, tuple[AnatomicalRegion, ...]] = {
    "ElbowLat": ("humeri", "radii", "ulnae"),
    "HipRay": ("femurs", "hips"),
    "LowerLimbs": ("femurs", "tibiae", "fibulae"),
}
DATASET_ID = "VictorButoi/flexray-data"


@pytest.fixture(scope="module")
def released_manifests():
    """Read only real-data manifests, excluding the synthetic FluXray database."""
    pytest.importorskip("fxr")
    yaml = pytest.importorskip("yaml")
    from huggingface_hub import hf_hub_download

    manifests = {}
    for dataset in DATASET_REGIONS:
        path = Path(
            hf_hub_download(DATASET_ID, f"{dataset}/dataset.yml", repo_type="dataset")
        )
        manifest = yaml.safe_load(path.read_text())
        assert manifest["dataset_name"] == dataset
        subjects = {
            split: {
                sample["subject_id"]
                for sample in manifest["samples"]
                if sample["split"] == split
            }
            for split in ("train", "val", "test")
        }
        assert not subjects["test"] & (subjects["train"] | subjects["val"]), (
            f"{dataset}: test subject IDs overlap another published partition"
        )
        # Resolve images and annotations from the same immutable data snapshot.
        manifests[dataset] = (manifest, path.parent.parent.name)
    return manifests


@pytest.mark.parametrize(
    ("dataset", "subject_index"),
    [
        ("ElbowLat", 0),
        ("ElbowLat", 1),
        ("HipRay", 0),
        ("HipRay", 1),
        ("LowerLimbs", 0),
        ("LowerLimbs", 2),
    ],
)
@save_returns("flexray_nonchest_annotations", unpack=True)
def test_flexray_released_anatomy(released_manifests, dataset: str, subject_index: int):
    """Save native-label Dice/IoU and absent-region false positives on held-out sources."""
    from huggingface_hub import hf_hub_download

    manifest, revision = released_manifests[dataset]
    unique_test_subjects = list(
        dict.fromkeys(
            sample["subject_id"]
            for sample in manifest["samples"]
            if sample["split"] == "test"
        )
    )
    subject_id = unique_test_subjects[subject_index]
    sample = next(
        sample
        for sample in manifest["samples"]
        if sample["split"] == "test" and sample["subject_id"] == subject_id
    )
    paths = {
        kind: Path(
            hf_hub_download(
                DATASET_ID,
                f"{dataset}/{sample[kind]}",
                repo_type="dataset",
                revision=revision,
            )
        )
        for kind in ("image", "label")
    }
    image = paths["image"].read_bytes()
    if paths["label"].suffix == ".npy":
        labels = np.load(paths["label"], allow_pickle=False)
        native_labels = manifest["label_names"]
        targets = {
            region: labels[native_labels.index(region)] > 0
            for region in DATASET_REGIONS[dataset]
        }
    else:
        with Image.open(paths["label"]) as label:
            labels = np.asarray(label)
        native_ids = {
            name: int(index) for index, name in manifest["stored_labels"].items()
        }
        targets = {
            region: labels == native_ids[region] for region in DATASET_REGIONS[dataset]
        }
    artifacts = {"source": image}
    measurements: list[dict] = []
    for region, target in targets.items():
        result = flexray_segment_anatomy(image, region=region)
        with Image.open(BytesIO(result)) as mask:
            prediction = np.asarray(mask) > 0
        assert prediction.shape == target.shape == (256, 256)
        intersection = int(np.count_nonzero(prediction & target))
        predicted, reference = int(prediction.sum()), int(target.sum())
        union = predicted + reference - intersection
        measurements.append(
            {
                "region": region,
                "reference_present": bool(reference),
                "reference_pixels": reference,
                "predicted_pixels": predicted,
                "false_positive_pixels": int(np.count_nonzero(prediction & ~target)),
                "dice": 2 * intersection / (predicted + reference)
                if predicted + reference
                else None,
                "iou": intersection / union if union else None,
            }
        )
        artifacts[region + "_prediction"] = result
        artifacts[region + "_reference"] = encode_as(
            Image.fromarray(target.astype(np.uint8) * 255)
        )
    assert any(measurement["reference_present"] for measurement in measurements)
    assert any(measurement["predicted_pixels"] > 0 for measurement in measurements), (
        f"{dataset}/{subject_id}: model returned no annotated anatomy"
    )
    return {
        **artifacts,
        "metrics": {
            "source": f"https://huggingface.co/datasets/{DATASET_ID}/tree/{revision}/{dataset}",
            "source_data_revision": revision,
            "dataset": dataset,
            "sample_id": sample["sample_id"],
            "subject_id": subject_id,
            "split": sample["split"],
            "manifest_subject_split_disjoint": True,
            "training_source_status": "excluded according to released flagship model card",
            "identity_note": "Subject IDs use the source's grouping; patient linkage across sources is not established",
            "geometry": "released real image/mask pairs, preprocessed to padded 256-square canvases",
            "model": "VictorButoi/flexray/members/flux0375, one pass, threshold 0.5",
            "regions": measurements,
        },
    }
