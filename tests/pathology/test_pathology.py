from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pytest
from PIL import Image

from caresmcp.api.decorators import matches_tool_signature, skip_undefined_kwargs
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import PATHOLOGY

SAMPLES = """
https://prod-images-static.radiopaedia.org/images/74884531/6f22f016116baa92e86b44a6bec617b490d615e54ebdbdc31078981193d6a675.png
https://prod-images-static.radiopaedia.org/images/74884532/8ed482a35146d868c83f7dd21128840039a515bbfa917c6831fdda93fbc3343f.png
https://prod-images-static.radiopaedia.org/images/74884533/4870726dffb5b8d84541c6e78c459f1051e33b30e4f689dfd33e6b2927621009.png
""".strip().splitlines()
PATHOLOGY_TOOLS = tuple(
    sorted(
        (name, function)
        for name, function in PATHOLOGY.get_all().items()
        if matches_tool_signature(
            function, return_type=bytes, available_inputs=("image",)
        )
    )
)


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize(
    "sample_url", SAMPLES, ids=[Path(urlparse(url).path).parent.name for url in SAMPLES]
)
@pytest.mark.parametrize(
    ("tool_name", "tool_function"),
    PATHOLOGY_TOOLS,
    ids=[name for name, _ in PATHOLOGY_TOOLS],
)
def test_pathology(
    tool_name: str,
    tool_function: Callable[..., object],
    sample_url: str,
) -> None:
    """Run pathology tools on image URLs and save outputs."""
    sample_path = Path(urlparse(sample_url).path)
    image = _read_bytes(sample_url)

    try:
        result = skip_undefined_kwargs(tool_function)(image=image)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"
    with Image.open(BytesIO(result)) as output_image:
        assert output_image.format == "PNG"
        assert output_image.mode == "RGB"
        with Image.open(BytesIO(image)) as input_image:
            assert output_image.size == input_image.size

    save_results(result, CACHE_ROOT / tool_name / sample_path.stem)
