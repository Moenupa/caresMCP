import base64
import binascii
from collections.abc import Callable
from inspect import signature
from os import PathLike
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints
from urllib.parse import unquote_to_bytes, urlsplit

from curl_cffi import requests

from caresmcp.constants import HTTP_TIMEOUT

from .base import adapt_inputs


def adapt_byte_inputs[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Adapt byte inputs, including nullable ones, from paths, data URIs, or URLs.

    Args:
        function: Function whose supplied byte-valued arguments should be adapted.

    Returns:
        The function wrapped to read byte inputs from supported string or path values.
    """

    byte_inputs = {name: _read_byte_input for name in _byte_input_names(function)}
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
    """Recognize byte sources and nullable byte-source collections."""
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in (Union, UnionType):
        variants = get_args(annotation)
        non_null = [variant for variant in variants if variant is not type(None)]
        return (
            type(None) in variants
            and len(non_null) == 1
            and _is_bytes_annotation(non_null[0])
        )
    if get_origin(annotation) is list:
        return get_args(annotation) == (bytes,)
    return annotation is bytes or annotation in ("bytes", "list[bytes]")


def _read_byte_input(value: Any) -> bytes | list[bytes] | None:
    """Read byte sources while preserving an omitted optional source."""
    if value is None:
        return None
    if isinstance(value, list):
        result = [_read_bytes(item) for item in value]
        if any(not item for item in result):
            raise ValueError("Byte input must not be empty")
        return result
    ret = _read_bytes(value)
    if not ret:
        raise ValueError("Byte input must not be empty")
    return ret


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
        # Match browser headers and TLS fingerprints for hosts that reject urllib.
        with requests.Session(impersonate="chrome") as session:
            try:
                response = session.get(
                    value, timeout=HTTP_TIMEOUT, allow_redirects=True
                )
                response.raise_for_status()
            except requests.RequestsError:
                # URLs may contain credentials or signed query parameters.
                raise ValueError("Unable to download HTTP byte input") from None
            return response.content

    with open(value, "rb") as source:
        return source.read()
