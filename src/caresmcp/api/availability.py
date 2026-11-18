"""Availability checks shared by the tool catalogue and the MCP server."""

import re
from collections.abc import Callable
from inspect import getdoc
from typing import Any

from caresmcp.api.decorators.deps import unmet_requirements

TODO_REGEX = [
    r"\bTODO\b",
    r"\btodo\b",
]


def docstring_summary(function: Callable[..., Any]) -> str:
    """Return the first non-empty line of a function docstring.

    Args:
        function: Callable to inspect.

    Returns:
        The summary line, or an empty string when the function has no docstring.
    """
    return next(
        (
            line.strip()
            for line in (getdoc(function) or "").splitlines()
            if line.strip()
        ),
        "",
    )


def has_todo(description: str) -> bool:
    """Return True if the description looks like a TODO item."""
    return any(re.search(regex, description) for regex in TODO_REGEX)


def unavailable_reasons(function: Callable[..., Any]) -> list[str]:
    """Return why a registered function cannot be used.

    A function is unavailable while its docstring summary is still a TODO
    placeholder, or while optional packages declared with
    :func:`caresmcp.api.decorators.requires` are not installed.

    Args:
        function: Registered callable to inspect.

    Returns:
        Human-readable reason clauses, empty when the function is usable.
    """
    reasons: list[str] = []
    if has_todo(docstring_summary(function)):
        reasons.append("its description is still a TODO placeholder")

    missing = unmet_requirements(function)
    if missing:
        noun = "package" if len(missing) == 1 else "packages"
        joined = ", ".join(f"'{package}'" for package in missing)
        reason = f"it requires the optional {noun} {joined}"
        extra = getattr(function, "__required_extra__", None)
        if extra:
            pronoun = "it" if len(missing) == 1 else "them"
            reason += f"; install {pronoun} with the '{extra}' extra"
        reasons.append(reason)

    return reasons
