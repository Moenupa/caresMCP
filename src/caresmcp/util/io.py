import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filetype import guess_extension


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
    """Print a text result or save a binary result to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path = path.with_suffix("." + (guess_extension(data) or "bin"))
        path.write_bytes(data)
        print_fn(f"[saved] Output to {path.resolve()}")
    else:
        # write as json
        path = path.with_suffix(".json")
        path.write_text(json.dumps(data, indent=2))
        print_fn(f"[saved] Output to {path.resolve()}")
        print_fn(f"[print] Output: {data}")
