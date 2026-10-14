"""Decorators for adapting values passed to utility functions."""

import base64
import binascii
from collections.abc import Callable
from functools import wraps
from inspect import Parameter, signature
from os import PathLike
from typing import Any, ParamSpec, TypeVar
from urllib.parse import unquote_to_bytes, urlsplit
from urllib.request import urlopen

P = ParamSpec("P")
R = TypeVar("R")


def replace_input(
    parameter: str,
    adapter: Callable[[Any], Any],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Transform one named argument before calling the decorated function.

    The argument can be supplied positionally or by keyword. If it is omitted,
    the wrapped function's default value is left untouched.

    Args:
        parameter: Name of the function argument to transform.
        adapter: Function that converts the supplied value to the expected form.

    Returns:
        A decorator that replaces the selected argument with its adapted value.

    Raises:
        ValueError: If ``parameter`` is not an argument of the decorated function.
    """

    if not parameter:
        raise ValueError("parameter must be a non-empty argument name")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        function_signature = signature(function)
        if parameter not in function_signature.parameters:
            function_name = getattr(function, "__name__", "callable")
            raise ValueError(f"{parameter!r} is not a parameter of {function_name}")

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = function_signature.bind_partial(*args, **kwargs)
            if parameter in bound.arguments:
                bound.arguments[parameter] = adapter(bound.arguments[parameter])
            return function(*bound.args, **bound.kwargs)

        return wrapped

    return decorate


def adapt_inputs(
    **adapters: Callable[[Any], Any],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Transform each supplied argument named in ``adapters``.

    This is a convenience form of :func:`replace_input` for functions that need
    more than one input conversion.

    Args:
        **adapters: Argument names mapped to their conversion functions.

    Returns:
        A decorator that applies each conversion before calling the function.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        function_signature = signature(function)
        unknown = set(adapters).difference(function_signature.parameters)
        if unknown:
            names = ", ".join(sorted(unknown))
            function_name = getattr(function, "__name__", "callable")
            raise ValueError(f"Unknown parameter(s) for {function_name}: {names}")

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = function_signature.bind_partial(*args, **kwargs)
            for name, adapter in adapters.items():
                if name in bound.arguments:
                    bound.arguments[name] = adapter(bound.arguments[name])
            return function(*bound.args, **bound.kwargs)

        return wrapped

    return decorate


def skip_undefined_kwargs[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Ignore keyword arguments that are not declared by ``function``.

    The wrapped callable still exposes the original signature to introspection,
    while accepting broader keyword dictionaries at runtime. This is useful
    when a caller shares one argument mapping across several related tools.

    Args:
        function: Callable whose undeclared keyword arguments should be ignored.

    Returns:
        The function wrapped to discard unknown keyword arguments.
    """

    function_signature = signature(function)
    if any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in function_signature.parameters.values()
    ):
        return function

    keyword_parameters = {
        name
        for name, parameter in function_signature.parameters.items()
        if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        filtered_kwargs = {
            name: value for name, value in kwargs.items() if name in keyword_parameters
        }
        return function(*args, **filtered_kwargs)  # ty: ignore[invalid-argument-type]

    return wrapped


def adapt_byte_inputs[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Adapt arguments annotated as ``bytes`` from paths, data URIs, or HTTP URLs.

    Args:
        function: Function whose supplied byte-valued arguments should be adapted.

    Returns:
        The function wrapped to read byte inputs from supported string or path values.
    """

    byte_inputs = {
        parameter.name: _read_bytes
        for parameter in signature(function).parameters.values()
        if parameter.annotation is bytes
    }
    if not byte_inputs:
        return function
    return adapt_inputs(**byte_inputs)(function)


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
