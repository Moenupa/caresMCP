"""FastMCP server wiring for catalogue-registered tools and prompts.

Run this module with ``fastmcp run src/caresmcp/mcpserver.py``.
"""

import warnings
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP

from caresmcp.api.decorators import adapt_fastmcp_tool
from caresmcp.api.decorators.deps import unmet_requirements
from caresmcp.registry import PROMPT, REGISTRIES


def _skip_for_missing_packages(
    kind: str, name: str, function: Callable[..., Any]
) -> bool:
    """Warn and return True when a registry entry lacks optional packages.

    Args:
        kind: Entry kind used in the warning, such as ``"tool"`` or ``"prompt"``.
        name: Registered entry name used in the warning.
        function: Registry function that may carry :func:`requires` metadata.

    Returns:
        True when the function requires packages that are not installed.
    """
    missing = unmet_requirements(function)
    if not missing:
        return False
    noun = "package" if len(missing) == 1 else "packages"
    joined = ", ".join(f"'{package}'" for package in missing)
    message = f"Skipping {kind} '{name}': requires the optional {noun} {joined}."
    extra = getattr(function, "__required_extra__", None)
    if extra:
        pronoun = "it" if len(missing) == 1 else "them"
        message += f" Install {pronoun} with the '{extra}' extra."
    warnings.warn(message, stacklevel=2)
    return True


mcp = FastMCP(name="caresmcp")


for category, registry in REGISTRIES.items():
    for name, function in registry.get_all().items():
        if _skip_for_missing_packages("tool", f"{category}.{name}", function):
            continue
        mcp.tool(name=name)(adapt_fastmcp_tool(function))

for name, function in PROMPT.get_all().items():
    if _skip_for_missing_packages("prompt", name, function):
        continue
    mcp.resource(uri=name if "://" in name else f"prompt://{name}")(function)
