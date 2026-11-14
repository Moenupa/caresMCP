"""Test byte input adaptation without external files or downloads."""

from inspect import signature
from pathlib import Path
from typing import Annotated
from unittest.mock import MagicMock, mock_open

import pytest

from caresmcp.api.decorators.bytes_io import adapt_byte_inputs


@pytest.fixture
def artifact() -> bytes:
    return b"fake artifact"


def test_byte_inputs(artifact, monkeypatch):
    """Accept bytes, file paths, data URIs, and HTTP URLs."""
    monkeypatch.setattr("builtins.open", mock_open(read_data=artifact))
    session = MagicMock()
    session.__enter__.return_value.get.return_value.content = artifact
    monkeypatch.setattr(
        "caresmcp.api.decorators.bytes_io.requests.Session", lambda **_: session
    )

    @adapt_byte_inputs
    def read(value: bytes) -> bytes:
        return value

    assert read(artifact) == artifact
    assert read(Path("artifact.bin")) == artifact  # ty: ignore[invalid-argument-type]
    assert read("artifact.bin") == artifact  # ty: ignore[invalid-argument-type]
    assert read("data:application/octet-stream;base64,ZmFrZSBhcnRpZmFjdA==") == artifact  # ty: ignore[invalid-argument-type]
    assert read("https://example.com/artifact.bin") == artifact  # ty: ignore[invalid-argument-type]


def test_lists_and_optional_inputs(artifact):
    """Adapt lists and optional byte inputs without changing other arguments."""

    @adapt_byte_inputs
    def read(
        value: Annotated[bytes | None, "artifact"] = None,
        values: list[bytes] | None = None,
        label: str = "example",
    ):
        return value, values, label

    data_uri = "data:;base64,ZmFrZSBhcnRpZmFjdA=="
    sources = [artifact, data_uri]
    assert read(data_uri, sources, label="test") == (  # ty: ignore[invalid-argument-type]
        artifact,
        [artifact, artifact],
        "test",
    )
    assert sources == [artifact, data_uri]
    assert read() == (None, None, "example")
    assert read(values=[]) == (None, [], "example")


def test_preserves_signature():
    def read(value: bytes, *, label: str = "example"):
        return value, label

    assert signature(adapt_byte_inputs(read)) == signature(read)


@pytest.mark.parametrize(
    "source, error",
    [
        (123, TypeError),
        ("data:text/plain,hello", ValueError),
        ("data:;base64,!!!", ValueError),
        ("https:///artifact.bin", ValueError),
    ],
)
def test_invalid_sources(source, error):
    @adapt_byte_inputs
    def read(value: bytes):
        return value

    with pytest.raises(error):
        read(source)
