"""Generate the task- and modality-oriented tool catalog pages.

The registry is the source of truth for tool names, categories, task tags, and
model choices. Run this script after changing the registry, then commit the
generated pages:

    uv run python scripts/generate_catalog.py
    uv run python scripts/generate_catalog.py --check
"""

import textwrap
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType
from typing import ClassVar, Self

import typer
from catalogue import Registry

import caresmcp  # noqa: F401  # importing populates the category registries
from caresmcp.api.availability import docstring_summary, has_todo
from caresmcp.api.io.annotation import unpack_literal
from caresmcp.registry import (
    REGISTRIES,
    TASK_CAP,
    TASK_CLS,
    TASK_DEN,
    TASK_EMB,
    TASK_EVAL,
    TASK_LOC,
    TASK_REG,
    TASK_RET,
    TASK_SEG,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
API_DIR = DOCS_DIR / "api"
CAPABILITIES_DIR = DOCS_DIR / "capabilities"
CATALOG_PATH = CAPABILITIES_DIR / "catalog.rst"
BY_MODALITY_PATH = CAPABILITIES_DIR / "by-modality.rst"

GENERATED_NOTICE = (
    "*Generated from the tool registry by ``scripts/generate_catalog.py``; "
    "do not edit by hand.*"
)
FOOTER_NOTE = (
    "See :doc:`../installation` for extras and :doc:`../quick-start` for "
    "the first call."
)

TABLE_CLASS = "capability-table"
CATALOG_TABLE_CLASS = f"{TABLE_CLASS} tool-catalog-table"
SUMMARY_TABLE_CLASS = f"{TABLE_CLASS} tool-summary-table"

TASK_SUMMARY_WIDTHS = (25, 10, 65)
CATALOG_TABLE_WIDTHS = (20, 20, 25, 35)
MODALITY_SUMMARY_WIDTHS = (25, 10, 25, 40)
MODALITY_TABLE_WIDTHS = (30, 20, 50)

# Table cells indent by seven columns (``   - - `` or ``       ``).
CELL_WIDTH = 81
# The closing note indents by three columns.
FOOTER_WIDTH = 85
PROSE_WIDTH = 88


def count_label(count: int) -> str:
    """Return a tool count with the correctly pluralized noun."""
    return f"{count} interface" if count == 1 else f"{count} interfaces"


def display_width(text: str) -> int:
    """Return the terminal display width of text containing emoji."""
    variation_selector = chr(0xFE0F)
    width = 0
    for index, character in enumerate(text):
        if character == variation_selector or unicodedata.combining(character):
            continue
        following = text[index + 1] if index + 1 < len(text) else ""
        if following == variation_selector or unicodedata.east_asian_width(
            character
        ) in {"W", "F"}:
            width += 2
        else:
            width += 1
    return width


@dataclass(frozen=True)
class Task:
    """A registry task and the catalog section it defines."""

    slug: str
    label: str
    output: str
    registry: Registry | None = field(default=None, compare=False, repr=False)
    empty: str = "No interface currently declares this task."

    @property
    def anchor(self) -> str:
        return f"capabilities-{self.slug}"

    @property
    def name(self) -> str:
        """Return the label without its leading emoji."""
        return self.label.split(" ", 1)[1]

    def heading(self, count: int) -> str:
        return f"{self.label} ({count_label(count)})"

    def registers(self, tool_name: str) -> bool:
        """Return whether this task's registry declares a tool."""
        return self.registry is not None and tool_name in self.registry.get_all()

    def matches(self, tool: "Tool") -> bool:
        """Return whether a tool belongs under this task's section."""
        if self.registry is None:
            return not tool.tasks
        return self in tool.tasks


TASKS = (
    Task("classify", "🎯 Classify", "Label probabilities or scores.", TASK_CLS),
    Task("regress", "📏 Regress", "Quantitative measurements.", TASK_REG),
    Task("segment", "🗺️ Segment", "Masks as PNG bytes.", TASK_SEG),
    Task("localize", "🔎 Localize", "Heatmaps, boxes, or ranked regions.", TASK_LOC),
    Task("retrieve", "🔁 Retrieve", "Similarity scores or rankings.", TASK_RET),
    Task("embed", "🧬 Embed", "Feature vectors.", TASK_EMB),
    Task("caption", "📝 Caption", "Text descriptions.", TASK_CAP),
    Task("denoise", "🧹 Denoise", "Filtered image bytes.", TASK_DEN),
    Task("evaluate", "🧮 Evaluate", "Reconstruction-error maps.", TASK_EVAL),
)
# The catalog page ends with the interfaces that declare no task.
OTHER_TASK = Task(
    "other",
    "🧰 Other interfaces",
    "Analysis and image utilities without a task tag.",
    empty="No interface currently falls in this category.",
)
SECTIONS = (*TASKS, OTHER_TASK)
TASK_POSITION = {task.slug: index for index, task in enumerate(TASKS)}


@dataclass(frozen=True)
class Category:
    """A registry category and the API page that documents it."""

    slug: str
    label: str
    api_page: str | None = None
    empty: ClassVar[str] = "No interface currently falls in this category."

    @property
    def page(self) -> str:
        """Return the API page stem for the category."""
        return self.api_page or self.slug

    @property
    def anchor(self) -> str:
        return f"capabilities-category-{self.slug}"

    def heading(self, count: int) -> str:
        return f"{self.label} ({count_label(count)})"


CATEGORIES = (
    Category("medai", "🤖 Medical AI"),
    Category("image", "📐 Generic Image Tools", "tools"),
    Category("ct", "🖥️ CT"),
    Category("dsa", "🫀 DSA and Catheter Angiography"),
    Category("mammography", "🩺 Mammography"),
    Category("mri", "🧲 MRI"),
    Category("pathology", "🔬 Pathology"),
    Category("ultrasound", "💓 Ultrasound"),
    Category("xray", "🩻 X-ray"),
)
CATEGORY_BY_SLUG = {category.slug: category for category in CATEGORIES}


def category_spec(slug: str) -> Category:
    """Return metadata for a category, defaulting to its slug as the label."""
    return CATEGORY_BY_SLUG.get(slug, Category(slug, slug))


@dataclass(frozen=True)
class Tool:
    """One servable tool and the metadata rendered in the catalog."""

    name: str
    category: Category
    tasks: tuple[Task, ...]
    models: tuple[str, ...]
    api: str
    description: str

    @classmethod
    def from_function(
        cls, category: Category, name: str, function: FunctionType
    ) -> "Tool | None":
        """Build a tool from a registered function, skipping TODO placeholders."""
        description = docstring_summary(function)
        if has_todo(description):
            return None
        return cls(
            name=name,
            category=category,
            tasks=tuple(task for task in TASKS if task.registers(name)),
            models=tuple(unpack_literal(function)),
            api=f"{function.__module__}.{function.__qualname__}",
            description=description,
        )


class Catalog:
    """Servable tools collected from the registry and grouped for rendering."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self.tools = tuple(sorted(tools, key=lambda tool: tool.name))
        self.categories = self._present_categories()
        self._by_task = {
            task.slug: tuple(tool for tool in self.tools if task.matches(tool))
            for task in SECTIONS
        }
        self._by_category = {
            category.slug: tuple(
                tool for tool in self.tools if tool.category.slug == category.slug
            )
            for category in self.categories
        }

    @classmethod
    def from_registry(cls) -> "Catalog":
        """Collect every servable tool from the category registries."""
        tools = []
        for slug, registry in REGISTRIES.items():
            category = category_spec(slug)
            for name, function in registry.get_all().items():
                tool = Tool.from_function(category, name, function)
                if tool is not None:
                    tools.append(tool)
        return cls(tools)

    def __len__(self) -> int:
        """Return the number of servable tools."""
        return len(self.tools)

    @property
    def category_count(self) -> int:
        """Return the number of categories with at least one tool."""
        return len(self.categories)

    def tools_in(self, task: Task) -> tuple[Tool, ...]:
        """Return the tools grouped under a task section."""
        return self._by_task[task.slug]

    def tools_in_category(self, category: Category) -> tuple[Tool, ...]:
        """Return the tools grouped under a category."""
        return self._by_category[category.slug]

    def _present_categories(self) -> tuple[Category, ...]:
        """Return categories with tools, in documented then alphabetical order."""
        present = {tool.category.slug: tool.category for tool in self.tools}
        known = [category for category in CATEGORIES if category.slug in present]
        extra = [
            present[slug] for slug in sorted(present.keys() - CATEGORY_BY_SLUG.keys())
        ]
        return (*known, *extra)


type Cell = Callable[[Tool], str]


def model_link(model: str) -> str:
    """Render a model choice, linking Hugging Face repository ids."""
    if "/" in model:
        return f"`{model} <https://huggingface.co/{model}>`_"
    return f"``{model}``"


def interface_cell(tool: Tool) -> str:
    """Render a tool name as a link to its API documentation."""
    return f":func:`{tool.name} <{tool.api}>`"


def category_cell(tool: Tool) -> str:
    """Render a tool's category as a link to its API page."""
    category = tool.category
    if (API_DIR / f"{category.page}.rst").exists():
        return f":doc:`{category.slug} <../api/{category.page}>`"
    return f"``{category.slug}``"


def models_cell(tool: Tool) -> str:
    """Render model choices, abbreviating lists longer than three."""
    if not tool.models:
        return "—"
    shown = ", ".join(model_link(model) for model in tool.models[:3])
    hidden = len(tool.models) - 3
    return f"{shown} … (+{hidden})" if hidden > 0 else shown


def task_labels(tasks: Iterable[Task]) -> str:
    """Render unique task names in catalog order, or an em dash when untagged."""
    ordered = sorted(set(tasks), key=lambda task: TASK_POSITION[task.slug])
    return ", ".join(task.name for task in ordered) or "—"


def tasks_cell(tool: Tool) -> str:
    return task_labels(tool.tasks)


def description_cell(tool: Tool) -> str:
    return tool.description


def representative_cell(tools: Sequence[Tool]) -> str:
    """Render links to up to three tools, or an em dash when there are none."""
    return ", ".join(interface_cell(tool) for tool in tools[:3]) or "—"


@dataclass(frozen=True)
class TableSpec:
    """The headers, column widths, and CSS class of an RST list-table."""

    headers: Sequence[str]
    widths: Sequence[int]
    class_name: str = TABLE_CLASS


class RstDocument:
    """Build a reStructuredText document block by block."""

    def __init__(self, width: int = PROSE_WIDTH) -> None:
        self.width = width
        self._blocks: list[str] = []

    def title(self, text: str, adornment: str = "=") -> None:
        """Append a title with a display-width-matched underline."""
        self._blocks.append(f"{text}\n{adornment * display_width(text)}")

    def paragraph(self, text: str, width: int | None = None) -> None:
        """Append wrapped prose."""
        self._blocks.append(
            textwrap.fill(
                text,
                width=width or self.width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    def heading(self, anchor: str, text: str) -> None:
        """Append an anchored section heading."""
        self._blocks.append(f".. _{anchor}:\n\n{text}\n{'-' * display_width(text)}")

    def table(self, spec: TableSpec, rows: Sequence[Sequence[str]]) -> None:
        """Append a list-table with a header row and the given body rows."""
        lines = [
            ".. list-table::",
            f"   :class: {spec.class_name}",
            "   :header-rows: 1",
            f"   :widths: {' '.join(str(width) for width in spec.widths)}",
            "",
        ]
        for row in [spec.headers, *rows]:
            for index, cell in enumerate(row):
                marker = "- -" if index == 0 else "  -"
                wrapped_cell = textwrap.wrap(
                    cell,
                    width=CELL_WIDTH,
                    break_long_words=False,
                    break_on_hyphens=False,
                ) or [""]
                lines.append(f"   {marker} {wrapped_cell[0]}")
                lines.extend(f"       {line}" for line in wrapped_cell[1:])
        self._blocks.append("\n".join(lines))

    def note(self, text: str) -> None:
        """Append a closing note directive."""
        body = textwrap.fill(
            text,
            width=FOOTER_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        ).replace("\n", "\n   ")
        self._blocks.append(f".. note::\n\n   {body}")

    def render(self) -> str:
        """Return the assembled document with a trailing newline."""
        return "\n\n".join(self._blocks) + "\n"


# Tasks and categories share the metadata a page section needs.
type Group = Task | Category


@dataclass(frozen=True)
class PageSection:
    """One rendered section: an anchor, heading, tools, and empty message."""

    anchor: str
    heading: str
    tools: tuple[Tool, ...]
    empty: str

    @classmethod
    def for_group(cls, group: Group, tools: Sequence[Tool]) -> Self:
        """Build the section for a task or category group."""
        return cls(group.anchor, group.heading(len(tools)), tuple(tools), group.empty)


class Page(ABC):
    """A generated page: an introduction, a summary table, and sections."""

    path: Path
    title: str
    summary: TableSpec
    table: TableSpec
    columns: Sequence[Cell]

    def render(self, catalog: Catalog) -> str:
        """Render the complete RST page."""
        rst = RstDocument()
        rst.title(self.title)
        rst.paragraph(GENERATED_NOTICE)
        rst.paragraph(self.intro(catalog))
        rst.table(self.summary, self.summary_rows(catalog))
        for section in self.sections(catalog):
            rst.heading(section.anchor, section.heading)
            if section.tools:
                rows = [
                    [column(tool) for column in self.columns] for tool in section.tools
                ]
                rst.table(self.table, rows)
            else:
                rst.paragraph(section.empty)
        rst.note(FOOTER_NOTE)
        return rst.render()

    @abstractmethod
    def intro(self, catalog: Catalog) -> str:
        """Return the page introduction."""

    @abstractmethod
    def summary_rows(self, catalog: Catalog) -> Sequence[Sequence[str]]:
        """Return the rows of the summary table."""

    @abstractmethod
    def sections(self, catalog: Catalog) -> Sequence[PageSection]:
        """Return the page sections in rendering order."""


class CatalogPage(Page):
    """The task-first catalog page."""

    path = CATALOG_PATH
    title = "Tool Catalog"
    summary = TableSpec(("Task", "Interfaces", "Typical output"), TASK_SUMMARY_WIDTHS)
    table = TableSpec(
        ("Interface", "Category", "Models", "Description"),
        CATALOG_TABLE_WIDTHS,
        CATALOG_TABLE_CLASS,
    )
    columns = (interface_cell, category_cell, models_cell, description_cell)

    def intro(self, catalog: Catalog) -> str:
        """Return the catalog page introduction."""
        return (
            f"CaresMCP registers **{count_label(len(catalog))}** across "
            f"**{catalog.category_count} categories**. An interface appears "
            "under each task it declares, so task counts overlap. Interfaces "
            "whose docstring is still a TODO placeholder are omitted; install "
            "the optional extra for an interface's backend before serving it "
            "over MCP."
        )

    def summary_rows(self, catalog: Catalog) -> list[list[str]]:
        """Return one summary row per task section."""
        return [
            [f":ref:`{task.anchor}`", str(len(catalog.tools_in(task))), task.output]
            for task in SECTIONS
        ]

    def sections(self, catalog: Catalog) -> list[PageSection]:
        """Return one section per task."""
        return [
            PageSection.for_group(task, catalog.tools_in(task)) for task in SECTIONS
        ]


class ModalityPage(Page):
    """The modality-first catalog page."""

    path = BY_MODALITY_PATH
    title = "Tools by Modality"
    summary = TableSpec(
        ("Category", "Interfaces", "Tasks", "Representative interfaces"),
        MODALITY_SUMMARY_WIDTHS,
        SUMMARY_TABLE_CLASS,
    )
    table = TableSpec(("Interface", "Tasks", "Description"), MODALITY_TABLE_WIDTHS)
    columns = (interface_cell, tasks_cell, description_cell)

    def intro(self, catalog: Catalog) -> str:
        """Return the modality page introduction."""
        return (
            f"CaresMCP registers **{count_label(len(catalog))}** across "
            f"**{catalog.category_count} categories**. Categories with no "
            "servable interfaces are documented in "
            ":doc:`../coverage/index`."
        )

    def summary_rows(self, catalog: Catalog) -> list[list[str]]:
        """Return one summary row per category."""
        return [self._summary_row(catalog, category) for category in catalog.categories]

    def sections(self, catalog: Catalog) -> list[PageSection]:
        """Return one section per category."""
        return [
            PageSection.for_group(category, catalog.tools_in_category(category))
            for category in catalog.categories
        ]

    def _summary_row(self, catalog: Catalog, category: Category) -> list[str]:
        tools = catalog.tools_in_category(category)
        return [
            f":ref:`{category.anchor}`",
            str(len(tools)),
            task_labels(task for tool in tools for task in tool.tasks),
            representative_cell(tools),
        ]


PAGES: tuple[Page, ...] = (CatalogPage(), ModalityPage())


def render_pages(catalog: Catalog, pages: Sequence[Page] = PAGES) -> dict[Path, str]:
    """Render every page to its output path."""
    return {page.path: page.render(catalog) for page in pages}


def write_pages(pages: dict[Path, str]) -> None:
    """Write rendered pages and report each path."""
    for path, content in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"Wrote {path.relative_to(ROOT)}")


def check_pages(pages: dict[Path, str]) -> None:
    """Exit with an error when a checked-in page differs from its rendering."""
    stale = [
        path
        for path, content in pages.items()
        if not path.exists() or path.read_text() != content
    ]
    if not stale:
        return
    print("Catalog pages are out of date; run `make catalog`.")
    for path in stale:
        print(f"  {path.relative_to(ROOT)}")
    raise typer.Exit(1)


def main(check: bool = False) -> None:
    """Write the catalog pages, or verify that checked-in copies are current."""
    pages = render_pages(Catalog.from_registry())
    if check:
        check_pages(pages)
    else:
        write_pages(pages)


if __name__ == "__main__":
    typer.run(main)
