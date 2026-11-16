from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def unavailable_packages(
    packages: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Return the import names that cannot be resolved.

    Args:
        packages: Import names to check, for example ``["torch", "timm"]``.

    Returns:
        Missing import names in the order they were given.
    """
    names = [packages] if isinstance(packages, str) else packages
    missing: list[str] = []
    for package in names:
        try:
            available = find_spec(package) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(package)
    return tuple(missing)


def unmet_requirements(function: Callable[..., Any]) -> tuple[str, ...]:
    """Return required optional packages that are unavailable for a function.

    Requirements are the packages attached by :func:`requires`. Functions that
    are not decorated with :func:`requires` have no unmet requirements.

    Args:
        function: Callable that may be decorated with :func:`requires`.

    Returns:
        Missing import names, empty when every requirement is installed.
    """
    return unavailable_packages(getattr(function, "__required_packages__", ()))


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
            missing_packages = unavailable_packages(normalized_packages)
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
