"""Pytest helpers for tests whose optional dependencies may be unavailable."""

from collections.abc import Callable
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, NoReturn, cast


def skip_on_import_error[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Skip a pytest test when it raises an ``ImportError``.

    Optional dependencies are checked at call time (for example by
    :func:`caresmcp.api.decorators.requires`), so a test for a tool can be
    decorated with this function to report a missing package as a skip instead
    of a failure. ``ModuleNotFoundError`` is handled because it subclasses
    ``ImportError``.

    Args:
        function: The test function or coroutine function to wrap.

    Returns:
        The wrapped function. Calling it raises pytest's skip outcome when an
        ``ImportError`` escapes the original call; other exceptions propagate.
    """

    if iscoroutinefunction(function):

        @wraps(function)
        async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
            __tracebackhide__ = True
            try:
                return await function(*args, **kwargs)
            except ImportError as exc:
                _skip(exc)

        return cast(Callable[P, R], async_wrapped)

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        try:
            return function(*args, **kwargs)
        except ImportError as exc:
            _skip(exc)

    return wrapped


def _skip(exc: ImportError) -> NoReturn:
    """Report ``exc`` as a pytest skip."""
    import pytest

    pytest.skip(str(exc) or type(exc).__name__)
