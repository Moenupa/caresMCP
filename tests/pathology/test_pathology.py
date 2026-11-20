"""Check Cellpose against the released CellposeModel on the official H&E tile.

The public tool returns a boundary overlay rather than raw masks, so the test
reruns the released ``CellposeModel.eval`` call on the same official CONCH
``docs/roi1.jpg`` tile and requires the overlay's boundaries, instance count,
and label box to reproduce the model's output exactly. The shared official
tile keeps the test minimal; Cellpose's benchmark archive is not downloaded.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

pytest.importorskip("cellpose")

from caresmcp.api.decorators import save_returns
from caresmcp.api.io import decode_as, encode_as
from caresmcp.constants import DEVICE
from caresmcp.pathology.cellpose import cellpose_segment_pathology_cells

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

# Center crop of the official tile, keeping the slow run bounded.
CROP_SIZE = 512

# Released tool defaults: automatic diameter with the Cellpose-SAM checkpoint.
EVAL_KWARGS = {
    "channel_axis": -1,
    "normalize": True,
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "min_size": 15,
}


@pytest.fixture(scope="module")
def he_tile(pathology_image: bytes) -> tuple[bytes, np.ndarray]:
    """Return the center of CONCH's official H&E tile as PNG bytes and pixels."""
    source = decode_as(pathology_image, mode="RGB")
    size = min(CROP_SIZE, source.width, source.height)
    left = (source.width - size) // 2
    top = (source.height - size) // 2
    tile = source.crop((left, top, left + size, top + size))
    return encode_as(tile), np.asarray(tile, dtype=np.uint8)


def _reference_masks(pixels: np.ndarray) -> np.ndarray:
    """Run the released Cellpose-SAM inference on the same pixels."""
    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=DEVICE != "cpu", pretrained_model="cpsam")
    masks, _, _ = model.eval(pixels, **EVAL_KWARGS)
    return np.asarray(masks)


def _reference_overlay(
    pixels: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Render the tool's documented overlay from upstream instance masks."""
    boundaries = np.zeros(masks.shape, dtype=bool)
    boundaries[1:, :] |= masks[1:, :] != masks[:-1, :]
    boundaries[:-1, :] |= masks[:-1, :] != masks[1:, :]
    boundaries[:, 1:] |= masks[:, 1:] != masks[:, :-1]
    boundaries[:, :-1] |= masks[:, :-1] != masks[:, 1:]
    boundaries &= masks > 0

    overlay = pixels.copy()
    if np.any(boundaries):
        boundary_color = np.array((255, 235, 59), dtype=np.float32)
        overlay[boundaries] = np.uint8(
            0.15 * overlay[boundaries].astype(np.float32) + 0.85 * boundary_color
        )

    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    count = int(np.unique(masks[masks > 0]).size)
    label = f"Cell instances: {count}"
    text_box = draw.textbbox((0, 0), label, font=font)
    padding = 4
    box = (
        4,
        4,
        text_box[2] - text_box[0] + padding * 2 + 4,
        text_box[3] - text_box[1] + padding * 2 + 4,
    )
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((box[0] + padding, box[1] + padding), label, fill=(255, 255, 255))
    return np.asarray(image), boundaries


@save_returns("pathology", unpack=True)
def test_cellpose_segment_pathology_cells(he_tile: tuple[bytes, np.ndarray]):
    """Reproduce the released model's boundaries, count, and label box."""
    image, pixels = he_tile
    result = cellpose_segment_pathology_cells(image)
    actual = np.asarray(decode_as(result, mode="RGB"))

    expected, boundaries = _reference_overlay(pixels, _reference_masks(pixels))
    assert boundaries.any(), "Expected cells in the official H&E tile"
    np.testing.assert_array_equal(actual, expected)
    return {"overlay": result}
