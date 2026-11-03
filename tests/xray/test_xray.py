"""Integration exercise for chest X-ray tools and sample image URLs."""

from collections.abc import Callable
from inspect import Parameter, signature
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from caresmcp.api.decorators import skip_undefined_kwargs
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import XRAY

SAMPLES = """
https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/blob/master/sample_xrays/Atelectasis.png?raw=true
https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/blob/master/sample_xrays/Cardiomegaly_Edema_Effusion.png?raw=true
https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/blob/master/sample_xrays/Effusion.png?raw=true
https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/blob/master/sample_xrays/Fibrosis.png?raw=true
https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/blob/master/sample_xrays/No%20Finding.png?raw=true
https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_000.png
""".strip().splitlines()
XRAY_TOOLS = [(k, v) for k, v in XRAY.get_all().items() if "embedding" not in k]


@pytest.mark.slow
@pytest.mark.parametrize(
    "sample_url",
    SAMPLES,
    ids=[unquote(Path(urlparse(url).path).name) for url in SAMPLES],
)
@pytest.mark.parametrize(
    ("tool_name", "tool_function"),
    XRAY_TOOLS,
    ids=[name for name, _ in XRAY_TOOLS],
)
def test_xray(
    tool_name: str,
    tool_function: Callable[..., object],
    sample_url: str,
) -> None:
    """Run compatible X-ray tools and save image results for inspection."""
    sample_path = Path(unquote(urlparse(sample_url).path))
    image = _read_bytes(sample_url)
    arguments: dict[str, object] = {
        "image": image,
        "saliency_map": image,
        "ground_truth_mask": image,
        "label": "Atelectasis",
        "unused_test_option": True,
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
    save_results(result, CACHE_ROOT / tool_name / sample_path.stem)
