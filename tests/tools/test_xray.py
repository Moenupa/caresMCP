"""Integration exercise for chest X-ray tools and bundled sample images."""

from collections.abc import Callable
from inspect import Parameter, signature
from pathlib import Path

import pytest

from caresmcp.registry import TAG_XRAY
from caresmcp.util.adapter import skip_undefined_kwargs
from caresmcp.util.io import save_results

SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "xray"
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
XRAY_TOOLS = tuple(sorted(TAG_XRAY.get_all().items()))


@pytest.mark.slow
@pytest.mark.parametrize(
    "sample_path", SAMPLE_PATHS, ids=[path.name for path in SAMPLE_PATHS]
)
@pytest.mark.parametrize(
    ("tool_name", "tool_function"),
    XRAY_TOOLS,
    ids=[name for name, _ in XRAY_TOOLS],
)
def test_xray(
    tool_name: str,
    tool_function: Callable[..., object],
    sample_path: Path,
) -> None:
    """Run compatible X-ray tools and save image results for inspection."""
    if any(keyword in tool_name for keyword in ("embedding", "saliency")):
        pytest.skip(f"Delibrately skipping {tool_name} in this test")

    print(f"Running {tool_name} on {sample_path.name}")
    image = sample_path.read_bytes()
    arguments: dict[str, object] = {
        "image": image,
        "saliency_map": image,
        "ground_truth_mask": image,
        "pathology": "Atelectasis",
        "sample_path": sample_path,
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
        pytest.skip("requires task-specific inputs: " + ", ".join(missing_arguments))

    try:
        result = skip_undefined_kwargs(tool_function)(**arguments)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert isinstance(result, (str, list, dict, bytes)), (
        f"Expected bytes, str, list, or dict, got {type(result)}"
    )
    save_results(result, Path(".cache") / tool_name / f"{sample_path.stem}")
