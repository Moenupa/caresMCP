"""Exercise promptable MedSAM and SAM2 segmentation on real medical images."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import decode_as, encode_as
from caresmcp.medai.hf import PromptableSegmentationModel, hf_segment_image


@pytest.fixture(scope="module")
def hsg_t_shaped_uterus() -> bytes:
    """T-shaped uterus on hysterosalpingography (Wikimedia, CC BY-SA 3.0)."""
    return _read_bytes(
        "https://upload.wikimedia.org/wikipedia/commons/8/83/T-shaped-uterus-001.jpg"
    )


@pytest.fixture(scope="module")
def rug_urethral_stricture() -> bytes:
    """Retrograde urethrogram with an annotated urethral stricture (CC BY 3.0)."""
    return _read_bytes(
        "https://upload.wikimedia.org/wikipedia/commons/5/5e/Urethra_stricture.jpg"
    )


@pytest.fixture(scope="module")
def vcug_vesicoureteral_reflux() -> bytes:
    """Voiding cystourethrogram with bilateral vesicoureteral reflux (CC BY-SA 3.0)."""
    return _read_bytes(
        "https://upload.wikimedia.org/wikipedia/commons/0/0a/Vesicoureteral-reflux-004.jpg"
    )


CASES = [
    pytest.param(
        "wanglab/medsam-vit-base",
        "hsg_t_shaped_uterus",
        (180.0, 120.0, 440.0, 500.0),
        id="medsam-hsg",
    ),
    pytest.param(
        "facebook/sam2.1-hiera-tiny",
        "vcug_vesicoureteral_reflux",
        (50.0, 50.0, 210.0, 310.0),
        id="sam2.1-tiny-vcug",
    ),
    pytest.param(
        "facebook/sam2.1-hiera-small",
        "rug_urethral_stricture",
        (300.0, 240.0, 590.0, 500.0),
        id="sam2.1-small-rug",
    ),
]


def _overlay(source: Image.Image, mask: np.ndarray) -> bytes:
    """Blend the binary mask over the source image for inspection."""
    pixels = np.asarray(source).copy()
    foreground = mask > 0
    pixels[foreground] = (
        pixels[foreground].astype(np.float32) * 0.5
        + np.array([255.0, 0.0, 0.0], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    return encode_as(Image.fromarray(pixels))


def _assert_mask(
    mask_bytes: bytes, source: Image.Image, box: tuple[float, ...]
) -> np.ndarray:
    """Verify original-size binary geometry and a prompted centroid."""
    with Image.open(BytesIO(mask_bytes)) as mask_image:
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


@pytest.mark.slow
@skip_on_import_error
@pytest.mark.parametrize(("model_id", "image_fixture", "box"), CASES)
@save_returns("hf_segment", unpack=True)
def test_hf_segment_image(
    model_id: PromptableSegmentationModel,
    image_fixture: str,
    box: tuple[float, float, float, float],
    request: pytest.FixtureRequest,
):
    """Return an original-size binary mask whose centroid is inside the prompt."""
    image = request.getfixturevalue(image_fixture)
    source = decode_as(image, mode="RGB")
    mask_bytes = hf_segment_image(image, box, model_id)
    mask = _assert_mask(mask_bytes, source, box)
    return {"source": image, "mask": mask_bytes, "overlay": _overlay(source, mask)}


@pytest.mark.slow
@pytest.mark.cuda
@skip_on_import_error
@save_returns("hf_segment", unpack=True)
def test_hf_segment_image_large(hsg_t_shaped_uterus: bytes):
    """Exercise the large SAM2.1 checkpoint when a CUDA device is available."""
    box = (180.0, 120.0, 440.0, 500.0)
    source = decode_as(hsg_t_shaped_uterus, mode="RGB")
    mask_bytes = hf_segment_image(
        hsg_t_shaped_uterus, box, "facebook/sam2.1-hiera-large"
    )
    mask = _assert_mask(mask_bytes, source, box)
    return {
        "source": hsg_t_shaped_uterus,
        "mask": mask_bytes,
        "overlay": _overlay(source, mask),
    }


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
    hsg_t_shaped_uterus: bytes,
    box: tuple[float, float, float, float],
):
    """Reject prompt boxes that are not finite and inside the image."""
    with pytest.raises(ValueError):
        hf_segment_image(hsg_t_shaped_uterus, box)
