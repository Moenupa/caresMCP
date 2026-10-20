"""Command-line entry point for listing registries and running the MCP server."""

import re
from itertools import zip_longest
from pathlib import Path

import seaborn as sns
import typer
from rich.console import Console
from rich.table import Table

from caresmcp.constants import CACHE_ROOT
from caresmcp.registry import REGISTRIES, TAGS, TASKS
from caresmcp.util.adapter import adapt_byte_inputs
from caresmcp.util.io import get_kwargs_from_file, save_results

app = typer.Typer(add_completion=False)
console = Console()

# Cycle of row styles assigned to categories (by sorted index)
COLORS = [
    "cyan",
    "yellow",
    "magenta",
    "green",
    "blue",
]
TAG_COLORS = {
    tag_name: color
    for tag_name, color in zip(
        sorted(TAGS),
        sns.color_palette("husl", n_colors=len(TAGS)).as_hex(),
    )
} | {
    tag_name: color
    for tag_name, color in zip(
        sorted(TASKS),
        sns.color_palette("husl", desat=0.5, n_colors=len(TASKS)).as_hex(),
    )
}


def _is_todo(description: str) -> bool:
    """Return True if the description looks like a TODO item."""
    return bool(re.search(r"\b[Tt][Oo][Dd][Oo]\b", description))


@app.command("list")
def list_functions(category: str | None = typer.Argument(None)) -> None:
    """Print registered tools and prompts grouped by catalogue category."""
    if category is not None and category not in REGISTRIES:
        valid_categories = ", ".join(REGISTRIES)
        raise typer.BadParameter(
            f"Choose one of: {valid_categories}", param_hint="category"
        )

    categories = [category] if category is not None else sorted(REGISTRIES)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Category")
    table.add_column("Name")
    table.add_column("Tags")
    table.add_column("Description")

    for cat_name, color in zip_longest(categories, COLORS):
        if cat_name is None:
            break

        registry = REGISTRIES[cat_name]
        table.add_section()
        for name, function in sorted(registry.get_all().items()):
            summary = (function.__doc__ or "").strip().splitlines()
            description = summary[0] if summary else ""
            tags = ", ".join(
                f"[white on {TAG_COLORS[tag_name]}] {tag_name.split("_")[-1]} [/]"
                for tag_name, tag_registry in sorted((TAGS | TASKS).items())
                if name in tag_registry.get_all()
            )
            table.add_row(
                f"[{color}]{cat_name}[/]",
                f"[{color}]{name}[/]",
                tags,
                description,
                style="dim" if _is_todo(description) else None,
            )

    console.print(table)


@app.command("debug")
def debug_function(category: str, name: str, kwarg_file: Path) -> None:
    """Run a registry function with JSON kwargs and adapted byte inputs.

    Byte inputs may be paths, base64 data URIs, or HTTP URLs.
    """
    kwargs = get_kwargs_from_file(kwarg_file)
    fn = REGISTRIES.get(category, {}).get(name)
    if fn is None:
        raise typer.BadParameter(
            f"Function '{name}' not found in category '{category}'",
            param_hint="name",
        )
    fn = adapt_byte_inputs(fn)

    for kwarg in kwargs:
        console.print(
            f"Debugging function {name!r} from category {category!r} with kwargs: {kwarg}"
        )
        output = fn(**kwarg)
        # if output is bytes, save to .cache
        save_results(output, CACHE_ROOT / category / f"{name}", console.print)


@app.command("mcp")
def mcpserve() -> None:
    """Start the FastMCP server."""
    from caresmcp.mcpserver import mcp

    mcp.run(transport="streamable-http", show_banner=False)
