"""Utilities for functions backed by optional Python packages."""

from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def get_torch_device() -> str:
    return get_cuda_device() or "cpu"


def get_cuda_device() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except ImportError:
        return None

    return None


def requires(
    packages: str | list[str],
    *,
    extra: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Require optional packages when a decorated function is called.

    The decorated module can be imported even when the package is unavailable.
    The dependency is checked at call time, which keeps optional tools from
    preventing the rest of the package from loading.

    Args:
        packages: Import names of the required packages (for example,
            ``["torch", "torchvision"]``). A string is also accepted for
            compatibility with single-package callers.
        extra: Optional installation extra to include in the error message.

    Returns:
        A decorator that checks package availability before calling the function.
    """

    package_names = [packages] if isinstance(packages, str) else packages
    if not isinstance(package_names, list) or not package_names:
        raise ValueError("packages must be a non-empty list of import names")

    normalized_packages: list[str] = []
    for package in package_names:
        if not isinstance(package, str) or not package.strip():
            raise ValueError("packages must contain non-empty import names")
        normalized = package.strip()
        if normalized not in normalized_packages:
            normalized_packages.append(normalized)

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            missing_packages: list[str] = []
            for package in normalized_packages:
                try:
                    available = find_spec(package) is not None
                except (ImportError, ModuleNotFoundError, ValueError):
                    available = False
                if not available:
                    missing_packages.append(package)
            if missing_packages:
                function_name = getattr(function, "__name__", "callable")
                joined_packages = ", ".join(
                    f"'{package}'" for package in missing_packages
                )
                noun = "package" if len(missing_packages) == 1 else "packages"
                message = (
                    f"{function_name} requires the optional {noun} {joined_packages}."
                )
                if extra:
                    message += f" Install them with the '{extra}' extra."
                raise ImportError(message)
            return function(*args, **kwargs)

        # Keep dependency information available to registries and tooling.
        wrapped.__dict__.update(
            __required_packages__=tuple(normalized_packages),
            __required_extra__=extra,
        )
        return wrapped

    return decorate
