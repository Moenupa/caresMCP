"""Pytest helpers for tests whose optional dependencies may be unavailable."""

from collections.abc import Callable
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, NoReturn, cast

type ExceptionTypes = (
    type[Exception] | list[type[Exception]] | tuple[type[Exception], ...]
)


def skip_on_exception[**P, R](
    exception_type: ExceptionTypes,
    *exception_types: ExceptionTypes,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Skip a pytest test when it raises one of the given exceptions.

    Optional dependencies can fail at call time with errors other than
    ``ImportError``. For example, a model download may raise ``OSError`` when
    the host is offline. Decorate the test with this function to report such
    conditions as skips instead of failures.

    Args:
        exception_type: An exception class that marks the test as skipped, or a
            list/tuple of such classes, for example ``OSError`` or
            ``[OSError, TimeoutError]``.
        *exception_types: Additional exception classes or lists/tuples of them.

    Returns:
        A decorator for a test function or coroutine function. Calling the
        wrapped function raises pytest's skip outcome when one of the given
        exceptions escapes the original call; other exceptions propagate.
        Subclasses of the given exception classes are also caught.
    """
    skip_types = tuple(
        item
        for group in (exception_type, *exception_types)
        for item in (group if isinstance(group, (list, tuple)) else (group,))
    )

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        if iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                __tracebackhide__ = True
                try:
                    return await function(*args, **kwargs)
                except skip_types as exc:
                    _skip(exc)

            return cast(Callable[P, R], async_wrapped)

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            __tracebackhide__ = True
            try:
                return function(*args, **kwargs)
            except skip_types as exc:
                _skip(exc)

        return wrapped

    return decorate


def skip_on_import_error[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Skip a pytest test when it raises an ``ImportError``.

    Optional dependencies are checked at call time (for example by
    :func:`caresmcp.api.decorators.requires`), so a test for a tool can be
    decorated with this function to report a missing package as a skip instead
    of a failure. ``ModuleNotFoundError`` is handled because it subclasses
    ``ImportError``. Use :func:`skip_on_exception` to skip on other errors.

    Args:
        function: The test function or coroutine function to wrap.

    Returns:
        The wrapped function. Calling it raises pytest's skip outcome when an
        ``ImportError`` escapes the original call; other exceptions propagate.
    """
    return skip_on_exception(ImportError)(function)


def _skip(exc: Exception) -> NoReturn:
    """Report ``exc`` as a pytest skip."""
    import pytest

    pytest.skip(str(exc) or type(exc).__name__)
