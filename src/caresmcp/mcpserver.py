"""FastMCP server wiring for catalogue-registered tools and prompts.

Run this module with ``fastmcp run src/caresmcp/mcpserver.py``.
"""

from fastmcp import FastMCP

from caresmcp.api.decorators import adapt_fastmcp_tool
from caresmcp.registry import PROMPT, REGISTRIES

mcp = FastMCP(name="caresmcp")


for registry in REGISTRIES.values():
    for name, function in registry.get_all().items():
        mcp.tool(name=name)(adapt_fastmcp_tool(function))

for name, function in PROMPT.get_all().items():
    mcp.resource(uri=name if "://" in name else f"prompt://{name}")(function)
