"""FastMCP server wiring for catalogue-registered tools and prompts.

Run this module with ``fastmcp run src/caresmcp/mcpserver.py``.
"""

import warnings
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP

from caresmcp.api.availability import unavailable_reasons
from caresmcp.api.decorators import adapt_fastmcp_tool
from caresmcp.registry import PROMPT, REGISTRIES


def _skip_if_unavailable(kind: str, name: str, function: Callable[..., Any]) -> bool:
    """Warn and return True when a registry entry cannot be served.

    Args:
        kind: Entry kind used in the warning, such as ``"tool"`` or ``"prompt"``.
        name: Registered entry name used in the warning.
        function: Registry function to inspect for incomplete implementations or
            missing optional packages.

    Returns:
        True when the function is unfinished or requires packages that are not
        installed.
    """
    reasons = unavailable_reasons(function)
    if not reasons:
        return False
    warnings.warn(f"Skipping {kind} '{name}': {'; '.join(reasons)}.", stacklevel=2)
    return True


mcp = FastMCP(name="caresmcp")


for category, registry in REGISTRIES.items():
    for name, function in registry.get_all().items():
        if _skip_if_unavailable("tool", f"{category}.{name}", function):
            continue
        mcp.tool(name=name)(adapt_fastmcp_tool(function))

for name, function in PROMPT.get_all().items():
    if _skip_if_unavailable("prompt", name, function):
        continue
    mcp.resource(uri=name if "://" in name else f"prompt://{name}")(function)
