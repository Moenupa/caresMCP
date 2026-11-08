import json
from collections.abc import Callable
from functools import wraps
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from filetype import guess_extension

from caresmcp.constants import CACHE_ROOT


def save_returns(
    collection: str,
    *,
    unpack: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., None]]:
    """Save a pytest function's return value under its parametrized test name.

    Use ``@save_returns("collection")`` on a test that returns JSON-serializable data
    or bytes. Results are saved to ``CACHE_ROOT / collection / pytest_node_name``
    with a JSON or detected binary extension.

    Tests may also declare the pytest ``request`` fixture themselves.
    With ``unpack=True``, return a dictionary of artifact names to values to save
    each value separately under the test's directory.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., None]:
        test_signature = signature(func)
        has_request = "request" in test_signature.parameters

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            request = kwargs["request"]
            if not has_request:
                kwargs.pop("request")
            result = func(*args, **kwargs)
            path = CACHE_ROOT / collection / request.node.name
            if unpack:
                for name, value in result.items():
                    artifact = path / name
                    save_results(value, artifact.with_name(artifact.name + ".result"))
                return
            # Reserve a suffix so save_results preserves dots in pytest node names.
            save_results(result, path.with_name(path.name + ".result"))

        if not has_request:
            parameters = list(test_signature.parameters.values())
            # A keyword-only fixture must precede a possible **kwargs parameter.
            index = next(
                (
                    i
                    for i, p in enumerate(parameters)
                    if p.kind == Parameter.VAR_KEYWORD
                ),
                len(parameters),
            )
            parameters.insert(index, Parameter("request", Parameter.KEYWORD_ONLY))
            wrapper.__dict__["__signature__"] = test_signature.replace(
                parameters=parameters
            )
        return wrapper

    return decorate


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
