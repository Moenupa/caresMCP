import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from filetype import guess_extension
from PIL import Image, ImageDraw, ImageFont


def get_kwargs_from_file(file: Path) -> list[dict]:
    kwargs = []
    if file.suffix.lower() == ".jsonl":
        for line in file.read_text().splitlines():
            kwargs.append(json.loads(line))
    elif file.suffix.lower() == ".json":
        kwargs.append(json.loads(file.read_text()))
    else:
        raise ValueError(f"Unsupported file type: {file.suffix}")
    return kwargs


def save_results(
    data: str | list | dict | bytes,
    path: Path,
    print_fn: Callable[[Any], None] = print,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path = path.with_suffix("." + (guess_extension(data) or "bin"))
        path.write_bytes(data)
    else:
        # write as json
        path = path.with_suffix(".json")
        path.write_text(json.dumps(data, indent=2))

    print_fn(f"[saved] Output to {path.resolve()}")


def draw_legends(
    image: Image.Image,
    labels: Sequence[str],
    *,
    title: str = "Legend",
    class_indices: Sequence[int] | None = None,
) -> Image.Image:
    """Add a color key beside a palette-based segmentation image.

    Args:
        image: Palette-based segmentation image.
        labels: Labels to show in the color key.
        title: Heading above the labels.
        class_indices: Palette indices matching ``labels``. Defaults to indices
            starting at 1.

    Returns:
        A palette-based image with the original image and its color key.
    """
    if image.mode != "P":
        raise ValueError("image must use palette mode")

    if class_indices is None:
        class_indices = range(1, len(labels) + 1)
    if len(labels) != len(class_indices):
        raise ValueError("labels and class_indices must have the same length")
    if any(not 0 <= index <= 255 for index in class_indices):
        raise ValueError("class_indices must be between 0 and 255")
    if len(set(class_indices)) != len(class_indices):
        raise ValueError("class_indices must be unique")

    palette = image.getpalette() or []
    palette.extend([0] * (768 - len(palette)))
    used_indices = set(image.getdata())  # ty: ignore[invalid-argument-type]
    free_indices = [
        index
        for index in range(256)
        if index not in used_indices and index not in class_indices
    ]
    if len(free_indices) < 2:
        raise ValueError("image palette must have two unused indices for the legend")

    background_index, text_index = free_indices[:2]
    palette[background_index * 3 : background_index * 3 + 3] = [255, 255, 255]
    palette[text_index * 3 : text_index * 3 + 3] = [0, 0, 0]

    font = ImageFont.load_default()
    measure = ImageDraw.Draw(Image.new("P", (1, 1)))
    text_width = max(
        measure.textbbox((0, 0), text, font=font)[2] for text in (title, *labels)
    )
    swatch_size = 14
    row_height = 20
    panel_padding = 12
    title_height = 20
    panel_width = int(panel_padding * 2 + swatch_size + 8 + text_width)
    legend_height = panel_padding * 2 + title_height + len(labels) * row_height

    source = image.copy()
    source.putpalette(palette)
    canvas = Image.new(
        "P",
        (source.width + panel_width, max(source.height, legend_height)),
        background_index,
    )
    canvas.putpalette(palette)
    canvas.info.update(image.info)
    canvas.paste(source, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line((source.width, 0, source.width, canvas.height), fill=text_index)
    legend_top = (canvas.height - legend_height) // 2
    panel_left = source.width
    draw.text(
        (panel_left + panel_padding, legend_top + panel_padding),
        title,
        fill=text_index,
        font=font,
    )

    rows_top = legend_top + panel_padding + title_height
    for row, (label, class_index) in enumerate(zip(labels, class_indices)):
        row_top = rows_top + row * row_height
        swatch_top = row_top + (row_height - swatch_size) // 2
        draw.rectangle(
            (
                panel_left + panel_padding,
                swatch_top,
                panel_left + panel_padding + swatch_size - 1,
                swatch_top + swatch_size - 1,
            ),
            fill=class_index,
            outline=text_index,
        )
        draw.text(
            (panel_left + panel_padding + swatch_size + 8, row_top + 4),
            label,
            fill=text_index,
            font=font,
        )

    return canvas
