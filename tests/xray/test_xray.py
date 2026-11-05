"""Integration exercise for chest X-ray tools and sample image URLs."""

from collections.abc import Callable
from inspect import Parameter, signature

import pytest

from caresmcp.api.decorators import skip_undefined_kwargs
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import XRAY

XRAY_TOOLS = [(k, v) for k, v in XRAY.get_all().items() if "embedding" not in k]


@pytest.fixture(scope="session", autouse=True)
def cxr_image() -> bytes:
    # a hernia case
    return _read_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_000.png"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("tool_name", "tool_function"),
    XRAY_TOOLS,
    ids=[name for name, _ in XRAY_TOOLS],
)
def test_xray(
    tool_name: str,
    tool_function: Callable[..., object],
    cxr_image: str,
) -> None:
    """Run compatible X-ray tools and save image results for inspection."""
    arguments: dict[str, object] = {
        "image": cxr_image,
        "label": "Atelectasis",
    }
    required_arguments = {
        name
        for name, parameter in signature(tool_function).parameters.items()
        if parameter.default is Parameter.empty
        and parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
    }
    missing_arguments = sorted(required_arguments.difference(arguments))
    if missing_arguments:
        pytest.skip("missing task-specific inputs: " + ", ".join(missing_arguments))

    try:
        result = skip_undefined_kwargs(tool_function)(**arguments)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert isinstance(result, (str, list, dict, bytes)), (
        f"Expected bytes, str, list, or dict, got {type(result)}"
    )
    save_results(result, CACHE_ROOT / tool_name)
