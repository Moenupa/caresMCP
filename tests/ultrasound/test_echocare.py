import math

import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.ultrasound.echocare import echocare_image_embed


@pytest.mark.slow
@pytest.mark.cuda
@save_returns("echocare_image_embed")
def test_echocare_with_real_ultrasound_image(ultrasound_image: bytes):
    """Extract an embedding from a real example when weights are available."""

    try:
        result = echocare_image_embed(ultrasound_image)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert len(result["embedding"]) == 2048
    assert all(math.isfinite(value) for value in result["embedding"])
    return result
