"""Exercise image processing on real images."""

from io import BytesIO
from typing import get_args

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.tools.denoise import DenoiseMethod, denoise


@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    get_args(DenoiseMethod),
)
@pytest.mark.parametrize("invariant", [False, True])
@save_returns("denoise")
def test_denoise(dummy_jpeg: bytes, method: DenoiseMethod, invariant: bool):
    """Denoise a real image and retain its dimensions and intensity scale."""
    pytest.importorskip("skimage")
    pytest.importorskip("pywt")
    result = denoise(image=dummy_jpeg, method=method, invariant=invariant)
    with Image.open(BytesIO(result)) as filtered:
        assert filtered.mode == "RGB"
        values = np.asarray(filtered)
        assert np.isfinite(values).all()
        assert values.min() >= 0 and values.max() <= 255
    return result
