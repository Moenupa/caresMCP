"""Exercise pediatric wrist detection on real annotated GRAZPEDWRI-DX images.

Images are the released 16-bit PNGs and annotations are the released YOLO
labels, fetched row by row from a revision-pinned mirror. Detections and their
overlap with source fracture annotations are checked against recorded values
from a deterministic CPU ONNX run, not only against generic box contracts.

Four representative projections cover fracture-positive and fracture-negative
cases across several acquisition angles and aspect ratios.
"""

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as, encode_as
from caresmcp.xray.wrist import WRIST_LABELS, detect_pediatric_wrist_findings

pytestmark = pytest.mark.slow

MODEL_ID = "mdciri/YOLOv7-Bone-Fracture-Detection"
CHECKPOINT = "trained-models/yolov7-p6-bonefracture.onnx"

# Recorded default-threshold detections in model output order from a
# deterministic CPU run. Labels and scores pin the ONNX graph and preprocessing;
# boxes are pinned separately by overlap with the source annotations below.
RECORDED_DETECTIONS: dict[str, tuple[tuple[str, float], ...]] = {
    "0032_1146492991_01_WRI-R1_M007": (("text", 0.9324), ("fracture", 0.86)),
    "0032_1146493016_02_WRI-R2_M007": (("text", 0.9138), ("fracture", 0.8252)),
    "0755_0509327622_01_WRI-R1_F012": (("text", 0.9016), ("fracture", 0.5029)),
    "2022_1185533267_01_WRI-L1_F012": (("text", 0.8573),),
}

# Best IoU between each source fracture annotation and its closest fracture
# detection from the same run. The negative case has no fracture annotations.
RECORDED_FRACTURE_IOUS: dict[str, tuple[float, ...]] = {
    "0032_1146492991_01_WRI-R1_M007": (0.8442,),
    "0032_1146493016_02_WRI-R2_M007": (0.5949,),
    "0755_0509327622_01_WRI-R1_F012": (0.6627,),
    "2022_1185533267_01_WRI-L1_F012": (),
}


def _box_iou(box: np.ndarray, target: np.ndarray) -> float:
    """Return IoU between normalized xyxy boxes."""
    extent = np.maximum(
        0, np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2])
    )
    intersection = float(np.prod(extent))
    union = float(
        np.prod(box[2:] - box[:2]) + np.prod(target[2:] - target[:2]) - intersection
    )
    return intersection / union


def _assert_recorded_detections(result: dict, image_id: str) -> None:
    """Check detection order, labels, and scores against the recorded output."""
    detections = result["detections"]
    expected = RECORDED_DETECTIONS[image_id]
    assert [detection["label"] for detection in detections] == [
        label for label, _ in expected
    ]
    assert [detection["score"] for detection in detections] == pytest.approx(
        [score for _, score in expected], abs=2e-3
    )


@skip_on_import_error
@save_returns("pediatric_wrist", unpack=True)
def test_detect_pediatric_wrist_findings(wrist_example):
    """Check box semantics and recorded fracture overlap on annotated projections."""
    image, row = wrist_example
    image_id = row["image_id"]
    result = detect_pediatric_wrist_findings(image)
    assert result["model_id"] == MODEL_ID
    assert result["checkpoint"] == CHECKPOINT
    width, height = result["image_width"], result["image_height"]
    assert (width, height) == (row["width"], row["height"])
    _assert_recorded_detections(result, image_id)
    assert result["detections"]
    assert any(detection["label"] == "fracture" for detection in result["detections"])
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
        overlaps = [
            _box_iou(
                np.asarray(detection["box_xyxy"]) / [width, height, width, height],
                target,
            )
            for detection in result["detections"]
            if detection["label"] == "fracture"
        ]
        fracture_ious.append(max(overlaps, default=0.0))
    assert fracture_ious and min(fracture_ious) > 0.1, (
        "Expected spatial overlap with these source fracture annotations"
    )
    assert fracture_ious == pytest.approx(RECORDED_FRACTURE_IOUS[image_id], abs=2e-3)
    return {
        "source": image,
        "annotations": row["objects"],
        "predictions": result,
        "fracture_best_ious": fracture_ious,
    }


@skip_on_import_error
@save_returns("pediatric_wrist_thresholds", unpack=True)
def test_detect_pediatric_wrist_findings_score_threshold(wrist_example):
    """Check score_threshold drops detections below the requested minimum."""
    image, _ = wrist_example
    default = detect_pediatric_wrist_findings(image)
    all_rows = detect_pediatric_wrist_findings(image, score_threshold=0.0)
    high = detect_pediatric_wrist_findings(image, score_threshold=0.9)
    none_pass = detect_pediatric_wrist_findings(image, score_threshold=1.0)
    for result in (default, all_rows, high, none_pass):
        assert result["model_id"] == MODEL_ID
    assert all_rows["detections"]
    assert none_pass["detections"] == []
    # The ONNX graph already drops low scores, so filtering only becomes
    # observable above the default; each fixture must keep and drop a row.
    assert 0 < len(high["detections"]) < len(all_rows["detections"])
    for threshold, result in ((0.3, default), (0.9, high)):
        assert result["detections"] == [
            detection
            for detection in all_rows["detections"]
            if detection["score"] >= threshold
        ]
    return {
        "all_rows": all_rows,
        "default": default,
        "threshold_0_9": high,
        "threshold_1": none_pass,
    }


@skip_on_import_error
def test_detect_pediatric_wrist_findings_16bit_png(wrist_example):
    """Check 16-bit PNGs keep the high byte used by the released OpenCV code."""
    image, _ = wrist_example
    source = decode_as(image)
    assert source.mode == "I;16"
    high_byte = (np.asarray(source).astype(np.uint16) >> 8).astype(np.uint8)
    scaled = encode_as(Image.fromarray(high_byte))
    result = detect_pediatric_wrist_findings(image)
    assert result["detections"]
    assert result == detect_pediatric_wrist_findings(scaled)


@skip_on_import_error
@save_returns("pediatric_wrist_cases", unpack=True)
def test_wrist_fracture_matching(wrist_case):
    """Measure fracture matches on one positive and one negative projection."""
    image, row = wrist_case
    image_id = row["image_id"]
    patient_id = str(row["patient_id"])

    result = detect_pediatric_wrist_findings(image)
    assert result["model_id"] == MODEL_ID
    width, height = result["image_width"], result["image_height"]
    assert (width, height) == (row["width"], row["height"])
    _assert_recorded_detections(result, image_id)
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
            iou = _box_iou(box, targets[index])
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= 0.5:
            remaining.remove(best_index)
            matched_ious.append(best_iou)
    true_positive = len(matched_ious)
    # Per-image regression expectations, not cohort performance estimates.
    assert true_positive == len(targets)
    assert len(predictions) == true_positive
    assert matched_ious == pytest.approx(RECORDED_FRACTURE_IOUS[image_id], abs=2e-3)
    return {
        "source": image,
        "annotations": row["objects"],
        "predictions": result,
        "metrics": {
            "patient_id": patient_id,
            "projection": row["projection"],
            "score_threshold": 0.3,
            "matching_iou_threshold": 0.5,
            "true_positive_fracture_boxes": true_positive,
            "false_positive_fracture_boxes": len(predictions) - true_positive,
            "false_negative_fracture_boxes": len(targets) - true_positive,
            "matched_ious": matched_ious,
        },
    }
