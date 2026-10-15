from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import TAG_PATHOLOGY
from caresmcp.util.adapter import skip_undefined_kwargs
from caresmcp.util.io import save_results

SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "pathology"
SAMPLE_PATHS = (
    tuple(
        sorted(
            path
            for path in SAMPLE_DIRECTORY.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    )
    if SAMPLE_DIRECTORY.exists()
    else ()
)
PATHOLOGY_TOOLS = tuple(sorted(TAG_PATHOLOGY.get_all().items()))


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize(
    "sample_path", SAMPLE_PATHS, ids=[path.name for path in SAMPLE_PATHS]
)
@pytest.mark.parametrize(
    ("tool_name", "tool_function"),
    PATHOLOGY_TOOLS,
    ids=[name for name, _ in PATHOLOGY_TOOLS],
)
def test_pathology(
    tool_name: str,
    tool_function: Callable[..., object],
    sample_path: Path,
) -> None:
    """Run pathology tools on bundled magnifications and save outputs."""
    print(f"Running {tool_name} on {sample_path.name}")
    image = sample_path.read_bytes()

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
