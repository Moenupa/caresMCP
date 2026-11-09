"""Convert Rich tables to tabular data."""

import pandas as pd
from rich.console import RenderableType
from rich.table import Table
from rich.text import Text


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
