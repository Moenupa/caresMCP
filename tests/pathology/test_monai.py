"""Real MONAI pathology inference using the shared pathology sample."""

from io import BytesIO
from typing import Any

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("monai")

from caresmcp.api.decorators import save_returns
from caresmcp.pathology.monai import (
    monai_pathology_nuclei_classification,
    monai_pathology_nuclei_segmentation_classification,
    monai_pathology_tumor_detection,
)

pytestmark = pytest.mark.slow


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def tissue_region(pathology_image: bytes) -> Image.Image:
    """Crop real tissue without changing the source scale."""
    with Image.open(BytesIO(pathology_image)) as source:
        x, y = source.width // 2, source.height // 2
        return source.convert("RGB").crop((x - 128, y - 96, x + 128, y + 96))


@pytest.fixture(scope="module")
def nuclei(tissue_region: Image.Image) -> dict[str, Any]:
    """Segment the real tissue region once for the nuclei tests."""
    return monai_pathology_nuclei_segmentation_classification(_png(tissue_region))


@save_returns("monai_pathology")
def test_tumor_patch_coordinates_and_batching(tissue_region: Image.Image):
    """Cover an ROI including padded edge patches with stable model scores."""
    image = _png(tissue_region)
    result = monai_pathology_tumor_detection(image, batch_size=1)
    batched = monai_pathology_tumor_detection(image, batch_size=4)
    assert result["image_size"] == {"width": 256, "height": 192}
    boxes = [[0, 0, 224, 192], [224, 0, 256, 192]]
    assert [patch["box"] for patch in result["patches"]] == boxes
    scores = [patch["tumor_score"] for patch in result["patches"]]
    assert all(0 <= score <= 1 for score in scores)
    assert scores == pytest.approx(
        [patch["tumor_score"] for patch in batched["patches"]], rel=1e-2
    )
    assert result["max_tumor_score"] == max(scores)
    return {"single": result, "batched": batched}


@save_returns("monai_pathology")
def test_hovernet_maps(nuclei: dict[str, Any]):
    """Return image-aligned instance and type maps with an explicit legend."""
    instances = np.asarray(nuclei["instance_map"])
    types = np.asarray(nuclei["type_map"])
    assert instances.shape == types.shape == (192, 256)
    assert instances.min() >= 0
    assert set(np.unique(types)) <= set(nuclei["labels"])
    assert nuclei["instance_count"] == len(np.unique(instances[instances > 0]))
    assert nuclei["instance_count"] > 0
    return nuclei


@save_returns("monai_pathology")
def test_nuclei_classifier(tissue_region: Image.Image, nuclei: dict[str, Any]):
    """Classify a real nucleus using its HoVerNet-predicted foreground mask."""
    instances = np.asarray(nuclei["instance_map"])
    positions = np.argwhere(instances > 0)
    assert len(positions), "Expected detected nuclei in the real tissue region"
    y, x = positions[len(positions) // 2]
    left = min(max(int(x) - 64, 0), tissue_region.width - 128)
    top = min(max(int(y) - 64, 0), tissue_region.height - 128)
    crop = tissue_region.crop((left, top, left + 128, top + 128))
    mask = instances[top : top + 128, left : left + 128] == instances[y, x]
    result = monai_pathology_nuclei_classification(
        _png(crop), _png(Image.fromarray(mask.astype(np.uint8) * 255))
    )
    scores = result["class_scores"]
    assert set(scores) == {"other", "inflammatory", "epithelial", "spindle_shaped"}
    assert all(0 <= value <= 1 for value in scores.values())
    assert sum(scores.values()) == pytest.approx(1.0)
    assert result["predicted_class"] == max(scores, key=scores.__getitem__)
    return result
