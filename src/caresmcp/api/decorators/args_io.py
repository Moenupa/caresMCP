from collections.abc import Callable
from functools import wraps
from inspect import Parameter, signature
from typing import Any


def matches_tool_signature(
    function: Callable[..., Any],
    *,
    return_type: type,
    available_inputs: tuple[str, ...],
) -> bool:
    """Check a tool's declared output type and required input names.

    Args:
        function: Tool to inspect.
        return_type: Exact return annotation required by the caller.
        available_inputs: Names of inputs the caller can supply.

    Returns:
        Whether the return annotation matches and every required parameter is
        available from the caller.
    """
    function_signature = signature(function)
    return function_signature.return_annotation is return_type and all(
        parameter.default is not Parameter.empty or parameter.name in available_inputs
        for parameter in function_signature.parameters.values()
    )


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
