import base64
import binascii
from collections.abc import Callable
from inspect import signature
from os import PathLike
from typing import Annotated, Any, get_args, get_origin, get_type_hints
from urllib.parse import unquote_to_bytes, urlsplit
from urllib.request import urlopen

from .base import adapt_inputs


def adapt_byte_inputs[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Adapt arguments annotated as ``bytes`` from paths, data URIs, or HTTP URLs.

    Args:
        function: Function whose supplied byte-valued arguments should be adapted.

    Returns:
        The function wrapped to read byte inputs from supported string or path values.
    """

    byte_inputs = {name: _read_bytes for name in _byte_input_names(function)}
    if not byte_inputs:
        return function
    return adapt_inputs(**byte_inputs)(function)


def _byte_input_names(function: Callable[..., Any]) -> tuple[str, ...]:
    """Return names of parameters annotated as bytes, including forward refs."""
    function_signature = signature(function)
    try:
        type_hints = get_type_hints(function, include_extras=True)
    except (NameError, TypeError):
        type_hints = {}

    return tuple(
        parameter.name
        for parameter in function_signature.parameters.values()
        if _is_bytes_annotation(type_hints.get(parameter.name, parameter.annotation))
    )


def _is_bytes_annotation(annotation: Any) -> bool:
    """Return whether an annotation declares bytes, including Annotated bytes."""
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation is bytes or annotation == "bytes"


def _read_bytes(value: Any) -> bytes:
    """Convert a byte input source into bytes."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, PathLike):
        with open(value, "rb") as source:
            return source.read()
    if not isinstance(value, str):
        raise TypeError("Byte inputs must be bytes, a path, a data URI, or an HTTP URL")

    parsed = urlsplit(value)
    if parsed.scheme.lower() == "data":
        metadata, separator, payload = value[5:].partition(",")
        if not separator or metadata.lower().split(";")[-1] != "base64":
            raise ValueError("Data URI byte inputs must use base64 encoding")
        try:
            return base64.b64decode(unquote_to_bytes(payload), validate=True)
        except binascii.Error as error:
            raise ValueError("Data URI contains invalid base64 data") from error

    if parsed.scheme.lower() in {"http", "https"}:
        if not parsed.netloc:
            raise ValueError("HTTP byte input must include a host")
        with urlopen(value, timeout=30) as response:
            return response.read()

    with open(value, "rb") as source:
        return source.read()
