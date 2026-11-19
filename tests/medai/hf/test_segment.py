"""Exercise promptable MedSAM and SAM2 segmentation on published demos."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import decode_as
from caresmcp.api.metrics import mask_overlap
from caresmcp.constants import DEVICE
from caresmcp.medai.hf import hf_segment_image

MEDSAM_REPO = "flaviagiammarino/medsam-vit-base"
MEDSAM_REVISION = "a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335"
DEMO_BOX = (95.0, 255.0, 190.0, 350.0)
MIN_DICE = 0.9

SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
SAM2_REVISION = "ee201e1a2ba9f944f26ab4b3219a32cf6dd27252"
SAM2_BOX = (425.0, 600.0, 700.0, 875.0)
SAM2_EXPECTED_IOU = 0.9689
SAM2_EXPECTED_LOGITS = ((-19.3144, -29.3947), (-24.6341, -24.1144))
MIN_IOU = 0.9
IOU_TOLERANCE = 0.01
LOGIT_TOLERANCE = 1e-2


def _assert_mask(
    mask_bytes: bytes, source: Image.Image, box: tuple[float, ...]
) -> np.ndarray:
    """Verify original-size binary geometry and a prompted centroid."""
    mask_image = decode_as(mask_bytes)
    assert mask_image.size == source.size
    assert mask_image.mode == "L"
    mask = np.asarray(mask_image)
    assert set(np.unique(mask)) <= {0, 255}
    foreground = mask == 255
    assert 0 < foreground.mean() < 0.5
    y_coordinates, x_coordinates = np.nonzero(foreground)
    assert box[0] <= x_coordinates.mean() <= box[2]
    assert box[1] <= y_coordinates.mean() <= box[3]
    return mask


def _published_demo_mask(output_png: bytes, size: tuple[int, int]) -> np.ndarray:
    """Extract the yellow mask overlay from the published MedSAM demo figure."""
    figure = np.asarray(decode_as(output_png, mode="RGB"))
    right = figure[:, figure.shape[1] // 2 :]
    nonwhite = np.any(right < 250, axis=2)
    columns = np.where(nonwhite.sum(axis=0) > 0.5 * nonwhite.shape[0])[0]
    rows = np.where(nonwhite.sum(axis=1) > 0.5 * nonwhite.shape[1])[0]
    plot = right[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    red, green, blue = (plot[..., index].astype(int) for index in range(3))
    yellow = (red + green) / 2 - blue > 30
    mask = Image.fromarray((yellow * 255).astype(np.uint8)).resize(
        size, Image.Resampling.NEAREST
    )
    return np.asarray(mask) > 0


@pytest.fixture(scope="session")
def medsam_demo() -> tuple[bytes, bytes]:
    """Published MedSAM demo image and rendered segmentation."""
    hub = pytest.importorskip("huggingface_hub")
    input_path = hub.hf_hub_download(
        MEDSAM_REPO, "scripts/input.png", revision=MEDSAM_REVISION
    )
    output_path = hub.hf_hub_download(
        MEDSAM_REPO, "scripts/output.png", revision=MEDSAM_REVISION
    )
    return Path(input_path).read_bytes(), Path(output_path).read_bytes()


@pytest.fixture(scope="session")
def sam2_truck() -> bytes:
    """Official SAM2 demo truck image from the Transformers test fixtures."""
    return _read_bytes(
        "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/"
        f"resolve/{SAM2_REVISION}/truck.jpg"
    )


@pytest.mark.slow
@skip_on_import_error
@save_returns("hf_segment_demo")
def test_medsam_reproduces_published_demo(
    medsam_demo: tuple[bytes, bytes],
) -> dict[str, Any]:
    """Reproduce the MedSAM model card's published demo segmentation.

    The card publishes the demo image and a rendered segmentation for box
    ``(95, 255, 190, 350)``. The yellow overlay is extracted from the right
    subplot as the expected mask; the prediction must reach Dice >= 0.9.
    """
    input_bytes, output_bytes = medsam_demo
    source = decode_as(input_bytes, mode="RGB")
    expected = _published_demo_mask(output_bytes, source.size)
    assert expected.any(), "Expected the published demo overlay to yield a mask"
    predicted = (
        np.asarray(decode_as(hf_segment_image(input_bytes, DEMO_BOX, MEDSAM_REPO))) > 0
    )
    dice = mask_overlap(predicted, expected).dice
    assert dice is not None and dice >= MIN_DICE, {"dice": dice}
    return {"model_id": MEDSAM_REPO, "box": list(DEMO_BOX), "dice": dice}


@pytest.mark.slow
@skip_on_import_error
@save_returns("hf_segment_demo")
def test_sam2_1_tiny_reproduces_published_inference(
    sam2_truck: bytes,
) -> dict[str, Any]:
    """Reproduce the published SAM2.1-tiny box-prompt inference values.

    Hugging Face's SAM2 integration suite records IoU 0.9689 and the top-left
    mask logits for box ``(425, 600, 700, 875)`` on the official demo truck
    image. The published values come from a four-box batch; this single-box
    path must match them as well, and the public tool must return an
    original-size binary mask whose centroid lies inside the prompt.

    See Also:
        https://github.com/huggingface/transformers/blob/v4.57.6/tests/models/sam2/test_modeling_sam2.py
    """
    import torch

    from caresmcp.api.models import HFModelType, get_hf_model_and_processor

    source = decode_as(sam2_truck, mode="RGB")
    model, processor = get_hf_model_and_processor(SAM2_MODEL_ID, HFModelType.SAM2)
    inputs = processor(
        images=source,
        input_boxes=[[[*SAM2_BOX]]],
        return_tensors="pt",
    )
    inputs = {
        name: value.to(DEVICE) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    assert outputs.iou_scores.shape == (1, 1, 1)
    assert outputs.pred_masks.shape == (1, 1, 1, 256, 256)
    iou = float(outputs.iou_scores[0, 0, 0])
    logits = outputs.pred_masks[0, 0, 0, :2, :2].float().cpu().numpy()
    assert iou >= MIN_IOU, {"iou": iou}
    assert abs(iou - SAM2_EXPECTED_IOU) <= IOU_TOLERANCE, {"iou": iou}
    assert np.allclose(logits, SAM2_EXPECTED_LOGITS, atol=LOGIT_TOLERANCE), {
        "logits": logits.tolist()
    }
    mask_bytes = hf_segment_image(sam2_truck, SAM2_BOX, SAM2_MODEL_ID)
    _assert_mask(mask_bytes, source, SAM2_BOX)
    return {"model_id": SAM2_MODEL_ID, "box": list(SAM2_BOX), "iou": iou}


@pytest.mark.parametrize(
    "box",
    [
        pytest.param((10.0, 10.0, 5.0, 50.0), id="reversed-x"),
        pytest.param((-1.0, 0.0, 10.0, 10.0), id="negative-x"),
        pytest.param((0.0, 0.0, 10_000.0, 10.0), id="outside-image"),
        pytest.param((0.0, 0.0, float("nan"), 10.0), id="non-finite"),
    ],
)
@skip_on_import_error
def test_hf_segment_image_rejects_invalid_box(
    dummy_jpeg: bytes,
    box: tuple[float, float, float, float],
):
    """Reject prompt boxes that are not finite and inside the image."""
    with pytest.raises(ValueError):
        hf_segment_image(dummy_jpeg, box)
