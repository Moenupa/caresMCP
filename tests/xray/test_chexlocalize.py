from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.xray.chexlocalize import chexlocalize_evaluate, chexlocalize_segment

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("smoothing_kernel_size", [0, 5])
@save_returns("chexlocalize")
def test_chexlocalize_segment(cxr_image: bytes, smoothing_kernel_size: int):
    """Check binary PNG outputs from real image intensities, without clinical claims."""
    result = chexlocalize_segment(
        cxr_image, smoothing_kernel_size=smoothing_kernel_size
    )
    with Image.open(BytesIO(cxr_image)) as source, Image.open(BytesIO(result)) as mask:
        assert mask.size == source.size
        assert mask.format == "PNG"
        assert mask.mode == "L"
        assert set(np.unique(np.asarray(mask))) == {0, 255}
    return result


@save_returns("chexlocalize")
def test_chexlocalize_evaluate(cxr_image: bytes):
    """Verify exact agreement with a mask derived from the same real image."""
    mask = chexlocalize_segment(cxr_image)
    result = chexlocalize_evaluate(cxr_image, mask)
    assert result == {"intersection_over_union": 1.0, "pointing_game_hit": 1.0}
    return result
