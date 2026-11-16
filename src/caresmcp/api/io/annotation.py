from collections.abc import Callable
from inspect import Parameter, formatannotation, signature, unwrap
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

MODEL_PARAMETER_NAMES = {
    "architecture",
    "encoder",
    "model",
    "model_id",
    "pretrained_model",
    "weights",
}


def format_annotation(annotation: Any) -> str:
    """Render an annotation without ``Annotated`` field metadata."""
    if annotation is Parameter.empty:
        return ""
    return formatannotation(_base_annotation(annotation))


def field_description(annotation: Any) -> str:
    """Return the pydantic ``Field`` description attached to an annotation."""
    if get_origin(annotation) is not Annotated:
        return ""
    for metadata in get_args(annotation)[1:]:
        description = getattr(metadata, "description", None)
        if isinstance(description, str) and description:
            return description
    return ""


def _base_annotation(annotation: Any) -> Any:
    """Strip runtime field metadata from an ``Annotated`` type."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


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
