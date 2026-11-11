"""Exercise image processing on real images."""

from io import BytesIO
from typing import Literal

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.tools.image_processing import denoise


@pytest.mark.slow
@pytest.mark.parametrize("method", ["bilateral", "gaussian", "median"])
@pytest.mark.parametrize("mode", ["L", "RGB"])
@save_returns("image_denoise", unpack=True)
def test_denoise(method: Literal["bilateral", "gaussian", "median"], mode: str):
    """Denoise a real image and retain its dimensions and intensity scale."""
    data = _read_bytes(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
    with Image.open(BytesIO(data)) as source:
        image = source.convert(mode)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    result = denoise(image=buffer.getvalue(), method=method)
    with Image.open(BytesIO(result)) as filtered:
        assert filtered.size == image.size
        assert filtered.mode == ("F" if mode == "L" else "RGB")
        values = np.asarray(filtered)
        assert np.isfinite(values).all()
        assert values.min() >= 0 and values.max() <= 255
    return {"source": buffer.getvalue(), "denoised": result}
