---
name: api-design
description: >-
  Decide what belongs in CaresMCP's shared api package. Use when writing tool
  logic, decoding inputs, loading a checkpoint, handling DEVICE or caches,
  adding a helper, or editing src/caresmcp/api/.
---

# What belongs in `api`

`caresmcp.api` exists to deduplicate boundary work: decoding inputs, moving
tensors to `DEVICE`, loading and caching checkpoints, encoding outputs. Without
it, every tool owns a slightly different copy, fixes miss tools, and identical
inputs behave differently. Deduplication is the criterion: share code when more
than one tool needs it, or when every tool must behave the same.

| Situation | Where it goes |
| --- | --- |
| More than one tool needs it | `api` |
| Must behave identically everywhere: conversion, `DEVICE`, caching, errors | `api` |
| New external weight source | `api/models/`, exported and cached there |
| One model or tool only | private `_helper` in the tool module |
| Several tools in one category | category package (`<category>/io.py`, `_arg.py`) |
| Unsure it will be reused | start private; promote when a second consumer appears |

Boundary conversions — caller bytes, DICOM, model loading, `DEVICE`, encoded
outputs — always go through `api`, even when a direct library call would be
shorter. Do not hide a direct call in a private helper to avoid the shared path.

When extending `api`, add the missing option to the shared helper instead of
forking it. Promote private code by moving it and updating callers, not by
copying it. A shared helper with one caller and no contract reason is just
indirection.

Function names and signatures live in `api/io/`, `api/models/`,
`api/decorators/`, and `caresmcp.constants`; their `__all__` and docstrings are
the source of truth.

## Dependency direction

Categories depend on `api`; `api` never imports a category. If a shared helper
needs category knowledge, it stays in the category.

## Verify

Run `make` after changing `api/`. Add a direct test when the helper's behavior
affects several tools. Tool authoring is covered by
`add-or-change-a-tool`.
