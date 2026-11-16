import json
from pathlib import Path
from typing import Any


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


def get_kwargs_from_str(s: str) -> list[dict]:
    """Parse inline JSON or JSON Lines text into a list of keyword arguments.

    Accepts a single JSON document (an object or a list of objects) or
    newline-delimited JSON with one object per non-empty line.
    """
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return [json.loads(line) for line in s.splitlines() if line.strip()]
    return data if isinstance(data, list) else [data]


def load_kwargs(source: str) -> list[dict[str, Any]]:
    """Load keyword arguments from inline JSON text or a JSON/JSONL file path.

    Inline text may hold a single JSON document or newline-delimited JSON.
    Anything else is treated as a path to a ``.json`` or ``.jsonl`` file.

    Raises:
        ValueError: If the source cannot be parsed or holds non-object entries.
        OSError: If a file source cannot be read.
    """
    source = source.strip()
    entries = (
        get_kwargs_from_str(source)
        if source.startswith(("{", "["))
        else get_kwargs_from_file(Path(source))
    )
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Each kwargs entry must be a JSON object")
    return entries
