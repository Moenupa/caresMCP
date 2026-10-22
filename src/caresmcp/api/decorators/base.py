from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any


def adapt_inputs[**P, R](
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
