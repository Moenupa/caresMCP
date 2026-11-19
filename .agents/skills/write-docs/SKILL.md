---
name: write-docs
description: >-
  Write or update docs/ or docstring. Use when editing APIs, tool docstrings,
  category survey docstrings, docs/api or docs/tools Sphinx pages, footcites,
  or when make docs fails.
---

# Write docs

Docs are Google-style docstrings rendered by Sphinx. `make docs` runs docstrfmt,
then `sphinx-build -T -W`; warnings fail the build. Never leave a parameter list
in a tool docstring: FastMCP treats it as duplicated input metadata and strips
it. Describe inputs in Pydantic `Field`s instead.

## Tool docstrings

- Open with one sentence on what the tool does, then explain how to interpret
  the output. Omit implementation details, input specs duplicated by `Field`s,
  self-explanatory output field lists, and when-to-use advice.
- Add sources and licenses whenever possible (models, repos, papers),
  keeping each reference next to what it supports.
- Cite papers in a standalone Google-style `References:` section with the full
  citation and a linked identifier (e.g.
  \`arXiv:2202.08238 <https://arxiv.org/abs/2202.08238>\`_). Don't use
  `:footcite:` in tool docstrings: `docs/api/` pages have no `footbibliography`,
  so the marker dangles. Keep repo/model links in `See Also:`.
- For `bytes` output, explain in docstring or legend the colors; single-color
  outputs can be explained in prose.

## Category survey (`src/caresmcp/<category>/__init__.py`)

The module docstring is the category survey; `docs/tools/<category>.rst` renders
it verbatim.

- Mark integrated tools with `|x|`, deferred candidates with `|o|`.
- Put deferred work under `Deferred priority candidates:` as `|o|` bullets and
  name the concrete blocker (license audit, gated weights, incompatible stack).
- Use `Out of scope:` only for candidates that cannot be reproduced at all — no
  released code, weights, or checkpoint. Deferred means TODO, not out of scope.
- One bullet per model family. Merge versioned siblings (Phikon/Phikon-v2,
  Virchow/Virchow2, UNI/UNI2-h) and state per-model differences inline; keep
  unrelated models separate.
- Cite one-to-one: put each model's `:footcite:` immediately after its link,
  never as a trailing collection of keys.
- Link the implementation with `:mod:`; state the output and access terms
  (gated, license, research-only).

Reference examples: `src/caresmcp/nuclearmed/__init__.py`,
`src/caresmcp/mammography/__init__.py`.

## Sphinx pages

- `docs/api/<category>.rst` - per-module `automodule`/`autofunction` listings
  for the category's tool functions.
- `docs/tools/<category>.rst` - tracking of implementation, status, links, citations.
- `docs/sources/*.rst` - model distributors/loaders, e.g. hf, torchhub.
- `docs/bib/<category>.bib` - per-category bib source to avoid git conflicts.

## Formatting gotchas

- Include an emoji at the start of a heading. Try to maintain usage consistency.
- `-W` fails the build on warnings: fix them, don't suppress.

## Verify

Run `make docs` and inspect `docs/_build/html`. To iterate without docstrfmt:
`uv run --extra docs sphinx-build -T -W -b html docs docs/_build/html`.
