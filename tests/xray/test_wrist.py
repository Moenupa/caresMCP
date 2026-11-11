"""Exercise pediatric wrist detection on real annotated GRAZPEDWRI-DX images."""

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.xray.wrist import WRIST_LABELS, detect_pediatric_wrist_findings

pytestmark = pytest.mark.slow


@save_returns("pediatric_wrist", unpack=True)
def test_detect_pediatric_wrist_findings(wrist_example):
    """Check box semantics and save source annotations alongside detections."""
    pytest.importorskip("onnxruntime")
    image, row = wrist_example
    result = detect_pediatric_wrist_findings(image)
    width, height = result["image_width"], result["image_height"]
    assert (width, height) == (row["width"], row["height"])
    assert result["detections"]
    assert any(d["label"] == "fracture" for d in result["detections"])
    for detection in result["detections"]:
        assert detection["label"] in WRIST_LABELS
        assert 0.3 <= detection["score"] <= 1
        left, top, right, bottom = detection["box_xyxy"]
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height
    fracture_ious = []
    for label, target in zip(
        row["objects"]["label"], row["objects"]["box"], strict=True
    ):
        if label != "fracture":
            continue
        target = np.asarray(target)
        overlaps = []
        for detection in result["detections"]:
            if detection["label"] != "fracture":
                continue
            box = np.asarray(detection["box_xyxy"]) / [width, height, width, height]
            extent = np.maximum(
                0, np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2])
            )
            intersection = float(np.prod(extent))
            union = float(
                np.prod(box[2:] - box[:2])
                + np.prod(target[2:] - target[:2])
                - intersection
            )
            overlaps.append(intersection / union)
        fracture_ious.append(max(overlaps, default=0.0))
    assert fracture_ious and min(fracture_ious) > 0.1, (
        "Expected spatial overlap with these two source fracture annotations"
    )
    return {
        "source": image,
        "annotations": row["objects"],
        "predictions": result,
        "fracture_best_ious": fracture_ious,
    }


@pytest.fixture(scope="module")
def published_wrist_splits():
    """Read the detector author's patient-separated train/validation/test manifests."""
    import csv

    root = (
        "https://raw.githubusercontent.com/mdciri/YOLOv7-Bone-Fracture-Detection/"
        "main/GRAZPEDWRI-DX_dataset/"
    )
    splits = {
        name: list(
            csv.DictReader(_read_bytes(root + name + "_data.csv").decode().splitlines())
        )
        for name in ("train", "valid", "test")
    }
    patients = {
        name: {row["patient_id"] for row in rows} for name, rows in splits.items()
    }
    assert not patients["train"] & patients["test"]
    assert not patients["valid"] & patients["test"]
    assert not patients["train"] & patients["valid"]
    return splits, patients


@pytest.mark.parametrize(
    ("offset", "image_id", "patient_id", "fracture_present"),
    [
        (18, "0755_0509327622_01_WRI-R1_F012", "755", True),
        (19, "0755_0509327645_01_WRI-R2_F012", "755", True),
        (79, "2022_1185533267_01_WRI-L1_F012", "2022", False),
        (80, "2022_1185533293_01_WRI-L2_F012", "2022", False),
    ],
)
@save_returns("pediatric_wrist_held_out", unpack=True)
def test_wrist_published_patient_split(
    published_wrist_splits,
    wrist_source_reader,
    offset: int,
    image_id: str,
    patient_id: str,
    fracture_present: bool,
):
    """Measure fracture box matches and false positives on two withheld patients."""
    pytest.importorskip("onnxruntime")
    splits, patients = published_wrist_splits
    assert patient_id in patients["test"]
    assert any(row["filestem"] == image_id for row in splits["test"])
    image, row = wrist_source_reader(offset, image_id)
    assert row["image_id"] == image_id
    assert str(row["patient_id"]) == patient_id
    assert ("fracture" in row["objects"]["label"]) is fracture_present
    result = detect_pediatric_wrist_findings(image)
    width, height = result["image_width"], result["image_height"]
    assert (width, height) == (row["width"], row["height"])
    targets = [
        np.asarray(box)
        for label, box in zip(
            row["objects"]["label"], row["objects"]["box"], strict=True
        )
        if label == "fracture"
    ]
    predictions = [
        np.asarray(detection["box_xyxy"]) / [width, height, width, height]
        for detection in result["detections"]
        if detection["label"] == "fracture"
    ]
    remaining = set(range(len(targets)))
    matched_ious = []
    for box in predictions:
        best_iou, best_index = 0.0, None
        for index in remaining:
            target = targets[index]
            extent = np.maximum(
                0, np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2])
            )
            intersection = float(np.prod(extent))
            union = float(
                np.prod(box[2:] - box[:2])
                + np.prod(target[2:] - target[:2])
                - intersection
            )
            iou = intersection / union
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_iou >= 0.5 and best_index is not None:
            remaining.remove(best_index)
            matched_ious.append(best_iou)
    true_positive = len(matched_ious)
    # Regression expectations for these four examples, not cohort performance.
    assert true_positive == len(targets)
    assert len(predictions) == true_positive
    return {
        "source": image,
        "annotations": row["objects"],
        "predictions": result,
        "metrics": {
            "patient_id": patient_id,
            "projection": row["projection"],
            "author_split": "test",
            "absent_from_author_train_and_validation_patients": True,
            "score_threshold": 0.3,
            "matching_iou_threshold": 0.5,
            "true_positive_fracture_boxes": true_positive,
            "false_positive_fracture_boxes": len(predictions) - true_positive,
            "false_negative_fracture_boxes": len(targets) - true_positive,
            "matched_ious": matched_ious,
            "cohort_note": "Four images, two patients; not a clinical performance estimate",
        },
    }
