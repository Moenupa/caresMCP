"""Run the coronary angiography U-Nets on a real ARCADE frame."""

import json

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as
from caresmcp.dsa import coronary

# Annotation for samples/126.png from the Stenoz release, as [x, y, width, height].
LESION_BOX_XYWH = (226.0, 71.62, 138.5, 145.26)


@pytest.mark.slow
@skip_on_import_error
@save_returns("dsa", unpack=True)
def test_detect_coronary_stenosis(coronary_angiogram: bytes):
    """Return scored candidate lesions and place the best inside the annotation."""
    result = coronary.detect_coronary_stenosis(coronary_angiogram)

    json.dumps(result, allow_nan=False)
    assert result["model_id"] == coronary.CORONARY_MODEL_ID
    assert result["checkpoint"] == "unet_stenosis.pt"
    assert result["image_width"] == 512
    assert result["image_height"] == 512
    assert result["detection_count"] == len(result["detections"])
    assert result["detection_count"] > 0
    assert 0 < result["max_detection_score"] <= 1
    for detection in result["detections"]:
        x, y = detection["centroid_xy"]
        assert 0 <= x < result["image_width"]
        assert 0 <= y < result["image_height"]
        assert 0 < detection["peak_score"] <= 1
        assert detection["area_pixels"] >= result["min_area_pixels"]

    left, top, width, height = LESION_BOX_XYWH
    x, y = result["detections"][0]["centroid_xy"]
    assert left <= x <= left + width
    assert top <= y <= top + height
    return {"detections": result}


@pytest.mark.slow
@skip_on_import_error
@save_returns("dsa", unpack=True)
def test_segment_coronary_stenosis(coronary_angiogram: bytes):
    """Return an original-size palette mask with candidate stenosis pixels."""
    result = coronary.segment_coronary_stenosis(coronary_angiogram)

    mask = decode_as(result)
    assert mask.size == (512, 512)
    assert mask.mode == "P"
    assert mask.info.get("transparency") == 0
    labels = np.asarray(mask)
    assert labels.max() == 1
    assert (labels == 1).sum() > 0
    return {"mask": result}


@pytest.mark.slow
@skip_on_import_error
@save_returns("dsa", unpack=True)
def test_segment_coronary_vessels(coronary_angiogram: bytes):
    """Return an original-size palette mask of the visible coronary tree."""
    result = coronary.segment_coronary_vessels(coronary_angiogram)

    mask = decode_as(result)
    assert mask.size == (512, 512)
    assert mask.mode == "P"
    labels = np.asarray(mask)
    assert labels.max() == 1
    assert (labels == 1).sum() > 0
    return {"mask": result}
