from collections.abc import Callable
from inspect import signature, unwrap
from typing import Any, Literal, get_args, get_origin, get_type_hints

MODEL_PARAMETER_NAMES = {
    "architecture",
    "encoder",
    "model",
    "model_id",
    "pretrained_model",
    "weights",
}


def unpack_literal(
    function: Callable[..., Any], target_argname: set[str] = MODEL_PARAMETER_NAMES
) -> list[str]:
    """Return declared model choices from Literal-annotated model parameters."""
    try:
        annotations = get_type_hints(unwrap(function))
    except (NameError, TypeError):
        annotations = {}

    model_values: list[str] = []
    for name, parameter in signature(function).parameters.items():
        if name not in target_argname:
            continue

        annotation = annotations.get(name, parameter.annotation)
        if get_origin(annotation) is Literal:
            model_values.extend(
                value for value in get_args(annotation) if isinstance(value, str)
            )
    return model_values
