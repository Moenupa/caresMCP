---
name: add-or-change-a-tool
description: >-
  Add, modify, or review a CaresMCP tool: its module, registration, signature,
  inputs, model choice, or output contract.
---

# Add or change a tool

Tools are plain functions in `src/caresmcp/<category>/<topic>.py`, registered
with `catalogue`. The category `__init__.py` auto-imports non-private modules,
so the decorators publish the tool. Keep it a thin wrapper: choose a model, run
the core step, return a self-describing result.

## Workflow

1. **Survey.** Read the category survey and neighboring modules; extend an
   existing tool when the change is a variant.
2. **Check servability.** Confirm released weights, an acceptable license, and
   a supported stack. Otherwise record a deferred `|o|` candidate with its
   concrete blocker.
3. **Place it.** One main category, every applicable task registry, and
   `TAG_MONAI` for MONAI. The snake_case function name is the public MCP/CLI
   name. Registry names are in `src/caresmcp/registry.py`.
4. **Declare the interface.** Use `Annotated[T, Field(...)]` for constraints
   and descriptions and `Literal[...]` for models and options. Image inputs are
   `bytes` named `image`, `dicom`, or `image_or_dicom`; adapters resolve paths,
   data URIs, and URLs. Default a single model; never rename a model ID.
5. **Implement.** Decode, load, move to `DEVICE`, and encode through
   `caresmcp.api` (see `api-design`); keep model-specific quirks private.
   Import optional packages inside the function and declare them in the
   matching `pyproject.toml` extra and `@requires(..., extra=...)`.
6. **Document and test.** Update the category survey and
   `docs/api/<category>.rst` (see `write-docs`); add a real-input test (see
   `write-tests`).
7. **Verify.** Run `make` and the focused test.

## Heuristics

- If a second tool would write the same line, it belongs in `api` or a
  decorator.
- Pydantic carries validation; never re-check or assert in the body. Raise
  `ValueError` for bad input and `RuntimeError` for model-contract violations,
  and keep messages free of URLs, tokens, and internal paths.
- Results explain themselves: meaningful keys, `model_id`, legends for colored
  bytes, and a one-sentence docstring with sources and licenses.
- No LLM/VLM or TensorFlow wrappers. A license or implementation blocker
  excludes a tool. CLIP can survive from this rule.

## Shape

```python
@XRAY.register("torchxrayvision_classify")  # one main category
@TASK_CLS.register("torchxrayvision_classify")  # every applicable task
@requires(["torchxrayvision"], extra="xray")
def torchxrayvision_classify(
    image_or_dicom: Annotated[
        bytes, Field(min_length=1, description="Chest radiograph or DICOM")
    ],
    weights: Annotated[
        Literal["densenet121-res224-all"],
        Field(description="TorchXRayVision checkpoint"),
    ] = "densenet121-res224-all",
) -> dict[str, Any]:
    """Score chest radiograph findings with TorchXRayVision.

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    ...
```

Use current modules (`src/caresmcp/pathology/conch.py`,
`src/caresmcp/xray/torchxrayvision.py`) as templates.
