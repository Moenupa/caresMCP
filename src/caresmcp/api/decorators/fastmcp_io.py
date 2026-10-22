"""Input and output conversions for FastMCP tool functions."""

import re
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any
from urllib.parse import urlsplit

import filetype
from fastmcp.utilities.types import File, Image

from .bytes_io import _byte_input_names, adapt_byte_inputs


def adapt_fastmcp_tool[**P, R](function: Callable[P, R]) -> Callable[P, Any]:
    """Adapt byte inputs and image outputs at the FastMCP tool boundary.

    Byte parameter names and annotations stay the same as the wrapped function.
    String inputs accepted by FastMCP are interpreted as HTTP(S) URLs or base64
    data URIs before the regular byte input adapter reads them.

    Args:
        function: Tool function registered with FastMCP.

    Returns:
        A callable with the same signature that adapts input sources and wraps
        byte results as FastMCP images or files.
    """
    function_signature = signature(function)
    byte_inputs = _byte_input_names(function)
    adapted_function = adapt_byte_inputs(function)

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
        bound = function_signature.bind_partial(*args, **kwargs)
        for name in byte_inputs:
            if name in bound.arguments:
                bound.arguments[name] = _normalize_mcp_byte_source(
                    bound.arguments[name]
                )

        result = adapted_function(*bound.args, **bound.kwargs)
        return _image_result(result)

    wrapped.__doc__ = _describe_mcp_inputs(function.__doc__, byte_inputs)
    return wrapped


def _normalize_mcp_byte_source(value: Any) -> bytes | str:
    """Convert FastMCP's validated bytes value back to a URL or data URI."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
        if not value.isprintable():
            return value.encode("utf-8")

    return _validate_image_source(value)


def _validate_image_source(value: Any) -> str:
    """Accept only HTTP(S) URLs and base64 data URIs for image inputs."""
    if not isinstance(value, str):
        raise TypeError("Image inputs must be an HTTP(S) URL or base64 data URI")

    parsed = urlsplit(value)
    if parsed.scheme.lower() == "data":
        return value
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return value
    raise ValueError("Image inputs must be an HTTP(S) URL or base64 data URI")


def _image_result(value: Any) -> Any:
    """Wrap image bytes in a FastMCP Image while preserving other results."""
    if not isinstance(value, bytes):
        return value

    file_type = filetype.guess(value)
    if file_type is None or not file_type.mime.startswith("image/"):
        return File(data=value, format=file_type.extension if file_type else None)
    return Image(data=value, format=file_type.extension)


def _describe_mcp_inputs(
    docstring: str | None,
    parameter_names: tuple[str, ...],
) -> str | None:
    """Describe URL and data URI inputs without changing parameter names."""
    if not docstring:
        return docstring

    for name in parameter_names:
        pattern = re.compile(rf"(?m)^([ \t]*){re.escape(name)}:[ \t]*(.*)$")
        docstring = pattern.sub(
            lambda match, name=name: (
                f"{match.group(1)}{name}: HTTP(S) URL or base64 data URI. "
                f"{match.group(2)}"
            ).rstrip(),
            docstring,
        )
    return docstring
