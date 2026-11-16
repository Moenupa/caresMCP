"""Build and convert Rich tables for display."""

from collections.abc import Callable
from functools import cache
from inspect import Parameter, getdoc, signature, unwrap
from typing import Any, get_type_hints

import pandas as pd
from rich import box
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from caresmcp.registry import REGISTRIES, TAGS, TASKS

from .annotation import field_description, format_annotation, unpack_literal
from .string import has_todo

# Cycle of row styles assigned to categories (by sorted index)
CATEGORY_COLORS = [
    "yellow",
    "cyan",
    "magenta",
    "green",
    "red",
    "blue",
] * 10


@cache
def _tag_colors() -> dict[str, str]:
    """Assign a distinct color to each tag and task on first use."""
    from seaborn import color_palette

    return {
        tag_name: color
        for tag_name, color in zip(
            sorted(TAGS), color_palette("husl", n_colors=len(TAGS)).as_hex()
        )
    } | {
        tag_name: color
        for tag_name, color in zip(
            sorted(TASKS),
            color_palette("husl", desat=0.5, n_colors=len(TASKS)).as_hex(),
        )
    }


def registry_table(category: str | None = None) -> Table:
    """Build a table of registered tools with models, tags, and descriptions.

    Args:
        category: Restrict the listing to one registered category.

    Raises:
        ValueError: If category is not registered.
    """
    if category is not None and category not in REGISTRIES:
        raise ValueError(f"Choose one of: {', '.join(REGISTRIES)}")

    table = Table(show_header=True, header_style="bold", box=box.MINIMAL_HEAVY_HEAD)
    table.add_column("Categ")
    table.add_column("Name")
    table.add_column("Models", max_width=30)
    table.add_column("Tags")
    table.add_column("Description", no_wrap=True)

    tag_colors = _tag_colors()
    for cat_name, color in zip(
        [category] if category is not None else sorted(REGISTRIES), CATEGORY_COLORS
    ):
        registry = REGISTRIES[cat_name]
        table.add_section()
        for name, function in sorted(registry.get_all().items()):
            summary = (function.__doc__ or "").strip().splitlines()
            description = summary[0] if summary else ""
            tags = ", ".join(
                f"[white on {tag_colors[tag_name]}]{tag_name.split('_')[-1]}[/]"
                for tag_name, tag_registry in sorted((TAGS | TASKS).items())
                if name in tag_registry.get_all()
            )
            table.add_row(
                f"[{color}]{cat_name}[/]",
                f"[{color}]{name}[/]",
                ",\n".join(unpack_literal(function)),
                tags,
                description,
                style="dim" if has_todo(description) else None,
            )
    return table


def function_help(function: Callable[..., Any]) -> Group:
    """Render a tool's docstring, parameters, defaults, and return type."""
    function = unwrap(function)
    try:
        hints = get_type_hints(function, include_extras=True)
    except (NameError, TypeError):
        hints = {}

    function_signature = signature(function)
    parameters = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    parameters.add_column("Parameter")
    parameters.add_column("Type", overflow="fold")
    parameters.add_column("Default", overflow="fold")
    parameters.add_column("Description")
    for parameter in function_signature.parameters.values():
        annotation = hints.get(parameter.name, parameter.annotation)
        default = (
            "[red]required[/]"
            if parameter.default is Parameter.empty
            else escape(repr(parameter.default))
        )
        parameters.add_row(
            escape(parameter.name),
            escape(format_annotation(annotation)),
            default,
            escape(field_description(annotation)),
        )

    sections: list[RenderableType] = []
    docstring = getdoc(function)
    if docstring:
        sections.append(Text(docstring))
    sections.append(parameters)
    returns = format_annotation(function_signature.return_annotation)
    if returns:
        sections.append(Text(f"Returns: {returns}"))
    return Group(*sections)


def _parse_renderable(value: RenderableType) -> RenderableType:
    """Extract plain text while preserving non-text renderables."""
    if isinstance(value, Text):
        return value.plain
    if isinstance(value, str):
        return Text.from_markup(value).plain
    return value


def rich_table_to_df(table: Table) -> pd.DataFrame:
    """Convert a Rich table to a DataFrame without display formatting.

    Preserve column order, duplicate headers, and empty rows and columns.
    Strip styles and Rich markup from Text and string headers and cells.
    Other renderables are retained as supplied. Footers, titles, and row
    separators are excluded.
    """
    columns = [_parse_renderable(column.header) for column in table.columns]
    data = {
        index: [_parse_renderable(cell) for cell in column.cells]
        for index, column in enumerate(table.columns)
    }
    frame = pd.DataFrame(data, index=range(table.row_count))
    frame.columns = columns
    return frame
