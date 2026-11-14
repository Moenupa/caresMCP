"""Measure pathology-label grounding against released NIH finding boxes.

Images are from the official NIH ChestX-ray14 image bucket; annotation bytes
are the NIH box release bundled with TorchXRayVision. BioViL model cards list
MIMIC-CXR as the image pretraining source. These are source-held-out checks,
not a calibrated disease-detection benchmark or a full-sentence grounding study.
"""

import csv
import gzip
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.models._biovil import BioViLModel
from caresmcp.xray.biovil import biovil_ground_phrases

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


@skip_on_import_error
@pytest.fixture(
    scope="module",
    params=[("00001688_000.png", "Nodule"), ("00019892_003.png", "Pneumothorax")],
)
def nih_box_example(request):
    """Read real same-finding boxes without inventing region annotations."""
    xrv = pytest.importorskip("torchxrayvision")
    filename, finding = request.param
    annotation_path = Path(xrv.__file__).parent / "data" / "BBox_List_2017.csv.gz"
    with gzip.open(annotation_path, "rt") as stream:
        rows = list(csv.reader(stream))[1:]
    boxes = []
    for row in rows:
        if row[0] == filename and row[1] == finding:
            left, top, width, height = map(float, row[2:6])
            boxes.append([left, top, left + width, top + height])
    assert boxes, f"No released {finding} annotation for {filename}"
    url = (
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/"
        + filename
    )
    image = _read_bytes(url)
    with Image.open(BytesIO(image)) as source:
        assert source.size == (1024, 1024), (
            "NIH box geometry is defined at 1024-square resolution"
        )
    return (
        image,
        boxes,
        finding,
        url,
        hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize(
    "model_id",
    ["microsoft/BiomedVLP-CXR-BERT-specialized", "microsoft/BiomedVLP-BioViL-T"],
)
@skip_on_import_error
@save_returns("biovil_nih_pathology_boxes", unpack=True)
def test_biovil_pathology_boxes(nih_box_example, model_id: BioViLModel):
    """Report peak hits and grid ranking inside the original-coordinate boxes."""
    image, boxes, finding, source_url, annotation_sha = nih_box_example
    result = biovil_ground_phrases(image, [finding], model_id=model_id)
    phrase = result["phrases"][0]
    scores = np.asarray(phrase["patch_similarity_map"])
    assert np.isfinite(scores).all()
    left, top, right, bottom = result["coverage_xyxy_original_pixels"]
    rows, columns = scores.shape
    xs = left + (np.arange(columns) + 0.5) * (right - left) / columns
    ys = top + (np.arange(rows) + 0.5) * (bottom - top) / rows
    target = np.zeros(scores.shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        target |= (
            (ys[:, None] >= y0)
            & (ys[:, None] < y1)
            & (xs[None, :] >= x0)
            & (xs[None, :] < x1)
        )
    assert target.any() and not target.all(), (
        "Expected both annotated and background grid cells"
    )
    foreground, background = scores[target], scores[~target]
    auc = float(
        (
            (foreground[:, None] > background)
            + 0.5 * (foreground[:, None] == background)
        ).mean()
    )
    peak_y, peak_x = np.unravel_index(scores.argmax(), scores.shape)
    return {
        "source": image,
        "grounding": result,
        "metrics": {
            "source_image": source_url,
            "source_annotation": "NIH BBox_List_2017.csv.gz bundled with TorchXRayVision",
            "annotation_sha256": annotation_sha,
            "source_patient_id": source_url.rsplit("/", 1)[1].split("_")[0],
            "model_training_image_source": "MIMIC-CXR according to model cards",
            "training_source_overlap": "NIH is not listed as an image pretraining source",
            "finding": finding,
            "boxes_xyxy_original_pixels": boxes,
            "peak_xy_original_pixels": [float(xs[peak_x]), float(ys[peak_y])],
            "peak_inside_released_box": bool(target[peak_y, peak_x]),
            "grid_box_auc": auc,
            "annotated_grid_cells": int(target.sum()),
            "background_grid_cells": int((~target).sum()),
            "global_cosine_similarity": phrase["global_cosine_similarity"],
            "negative_note": "Outside-box grid cells are localization background, not verified healthy tissue",
        },
    }
