import math
from pathlib import Path

import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.registry import TASK_EMBED, ULTRASOUND
from caresmcp.ultrasound.echocare import echocare_image_embed

SAMPLES = """
https://prod-images-static.radiopaedia.org/images/74884529/81c7389cb9f3846ae942256a998303cf4eca7b110e35a3749e60b96aebc0cdd0.png
https://prod-images-static.radiopaedia.org/images/74884530/ae784ed9410c42c13af06381284a00e541ef4f293f7c57a9afc4eafff93ff643.png
""".strip().splitlines()


def test_echocare_tool_is_registered() -> None:
    """Expose EchoCare image embeddings through both catalogues."""
    assert "echocare_image_embed" in ULTRASOUND.get_all()
    assert "echocare_image_embed" in TASK_EMBED.get_all()


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize(
    "sample_url",
    SAMPLES,
    ids=[Path(url).parent.stem for url in SAMPLES],
)
@save_returns("echocare_image_embed")
def test_echocare_with_real_ultrasound_image(sample_url: str):
    """Extract an embedding from a real example when weights are available."""
    image = _read_bytes(sample_url)

    try:
        result = echocare_image_embed(image)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert len(result["embedding"]) == 2048
    assert all(math.isfinite(value) for value in result["embedding"])
    return result
