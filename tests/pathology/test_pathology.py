from collections.abc import Callable
from io import BytesIO

import pytest
from PIL import Image

from caresmcp.api.decorators import matches_tool_signature, skip_undefined_kwargs
from caresmcp.api.decorators.file_io import save_results
from caresmcp.api.imageio import decode_rgb
from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import PATHOLOGY

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
    ("tool_name", "tool_function"),
    PATHOLOGY_TOOLS,
    ids=[name for name, _ in PATHOLOGY_TOOLS],
)
def test_pathology(
    tool_name: str,
    tool_function: Callable[..., object],
    pathology_image: bytes,
) -> None:
    """Run pathology tools on image URLs and save outputs."""
    try:
        result = skip_undefined_kwargs(tool_function)(image=pathology_image)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"
    with Image.open(BytesIO(result)) as output_image:
        assert output_image.format == "PNG"
        assert output_image.mode == "RGB"
        input_image = decode_rgb(pathology_image)
        assert output_image.size == input_image.size

    save_results(result, CACHE_ROOT / tool_name)
