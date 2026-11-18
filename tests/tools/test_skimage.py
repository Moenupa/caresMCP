"""Exercise image processing on real images."""

from typing import get_args

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as
from caresmcp.tools.skimage import DenoiseMethod, denoise


@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    get_args(DenoiseMethod),
)
@pytest.mark.parametrize("invariant", [False, True])
@skip_on_import_error
@save_returns("denoise")
def test_denoise(dummy_jpeg: bytes, method: DenoiseMethod, invariant: bool):
    """Denoise a real image and retain its dimensions and intensity scale."""
    result = denoise(image=dummy_jpeg, method=method, invariant=invariant)
    filtered = decode_as(result)
    assert filtered.mode == "RGB"
    values = np.asarray(filtered)
    assert np.isfinite(values).all()
    assert values.min() >= 0 and values.max() <= 255
    return result
