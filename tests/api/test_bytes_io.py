"""Exercise byte input adaptation with a real medical image."""

import base64
from inspect import signature
from pathlib import Path
from typing import Annotated, Any

import pytest

from caresmcp.api.decorators.bytes_io import adapt_byte_inputs

pytestmark = pytest.mark.slow

IMAGE_URL = "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_000.png"
type ImageSource = bytes | Path | str


@pytest.fixture(scope="module")
def image_sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, ImageSource]:
    """Provide supported sources for the same real medical image."""

    @adapt_byte_inputs
    def read_image(image: bytes) -> bytes:
        return image

    image = read_image(IMAGE_URL)  # ty: ignore[invalid-argument-type]
    path = tmp_path_factory.mktemp("byte-inputs") / "image.png"
    path.write_bytes(image)
    return {
        "bytes": image,
        "path": path,
        "path_string": str(path),
        "data_uri": "data:image/png;base64," + base64.b64encode(image).decode("ascii"),
        "url": IMAGE_URL,
    }


@pytest.mark.parametrize(
    "source_format, annotation",
    [
        ("bytes", bytes),
        ("path", Annotated[bytes, "image"]),
        ("path_string", "bytes"),
        ("data_uri", bytes),
        ("url", bytes),
    ],
)
def test_single_byte_input(annotation: Any, source_format, image_sources):
    """Adapt each source and preserve the signature and other arguments."""

    def tool(value, *, label: str = "image"):
        return value, label

    tool.__annotations__["value"] = annotation
    adapted = adapt_byte_inputs(tool)
    source = image_sources[source_format]

    assert signature(adapted) == signature(tool)
    assert adapted(source, label="scan") == (image_sources["bytes"], "scan")
    assert adapted(value=source) == (image_sources["bytes"], "image")


@pytest.mark.parametrize(
    "annotation", [list[bytes], Annotated[list[bytes], "images"], "list[bytes]"]
)
def test_byte_input_list(annotation: Any, image_sources):
    """Adapt source lists without changing the caller's list."""

    def tool(values):
        return values

    tool.__annotations__["values"] = annotation
    adapted = adapt_byte_inputs(tool)
    sources = list(image_sources.values())
    original_sources = sources.copy()

    expected = [image_sources["bytes"]] * len(sources)

    assert adapted(sources) == expected
    assert adapted(values=sources) == expected
    assert sources == original_sources
    assert adapted([]) == []


def test_multiple_byte_parameters(image_sources):
    """Adapt single and list byte parameters together at the public boundary."""

    @adapt_byte_inputs
    def tool(image: bytes, references: list[bytes], path: Path):
        return image, references, path

    path = Path("unchanged")
    sources = list(image_sources.values())
    assert tool(image_sources["path"], references=sources, path=path) == (
        image_sources["bytes"],
        [image_sources["bytes"]] * len(sources),
        path,
    )
