🤝 Contributing
===============

CaresMCP is built from small, registered tools and one model survey per category. This
page covers the development loop for both.

🛠️ Set Up the Development Environment
-------------------------------------

.. code-block:: console

   $ git clone https://github.com/Moenupa/caresMCP.git
   $ cd caresMCP
   $ uv sync --all-extras
   $ make          # format, lint, and type check

Run the focused test suite with ``make test`` and the full one with ``make test-slow``.
A single test file can be run directly:

.. code-block:: console

   $ uv run pytest tests/xray/test_torchxrayvision.py

Use ``feat/xxx``, ``fix/xxx``, or ``doc/xxx`` for branch names.

➕ Add a Tool
-------------

1. Add the function to the relevant module under ``src/caresmcp/<category>/``.
2. Register it with the category registry and every task it serves:

   .. code-block:: python

      @XRAY.register("my_tool")
      @TASK_CLS.register("my_tool")
      def my_tool(image: Annotated[bytes, IMAGE_FIELD]) -> dict:
          """Score a finding in one image."""

3. Describe each input with a Pydantic ``Field``. Do not put a parameter list in the
   docstring: FastMCP treats it as duplicated input metadata.
4. Keep the docstring to one sentence on what the tool does, then explain how to
   interpret the output. Put full citations in a Google-style ``References:`` section
   and repository or model links in ``See Also:``.
5. Add a test under ``tests/<category>/`` that exercises the public contract.
6. Add the function to ``docs/api/<category>.rst`` with ``autofunction``.
7. Regenerate the capability catalog:

   .. code-block:: console

      $ make catalog

   Commit the regenerated ``docs/capabilities/catalog.rst`` and
   ``docs/capabilities/by-modality.rst`` with the code change.

🔬 Add or Update a Model Survey
-------------------------------

Each category's module docstring is its survey, rendered under :doc:`coverage/index`.

- Mark integrated models with ``|x|`` and deferred candidates with ``|o|``.
- Put deferred work under ``Deferred priority candidates:`` and name the concrete
  blocker (license audit, gated weights, incompatible stack).
- Use ``Out of scope:`` only for candidates with no released code, weights, or
  reproducible path.
- One bullet per model family. Merge versioned siblings and state per-model differences
  inline.
- Cite one-to-one: put each model's ``:footcite:`` immediately after its link.
- Add the bib entry to ``docs/bib/<category>.bib``.
- Link the implementation with ``:mod:`` and state output and access terms (gated,
  license, research-only).

📚 Rebuild the Docs
-------------------

.. code-block:: console

   $ make docs

This formats the docs and builds them with warnings as errors. To iterate without
formatting:

.. code-block:: console

   $ uv run --extra docs sphinx-build -T -W -b html docs docs/_build/html

The capability pages are generated from the registry. ``make catalog-check`` verifies
that the checked-in copies are current; run it before opening a pull request if you
changed a registry entry.

🧪 Tests and Formatting
-----------------------

- ``make format`` — docstrfmt, Ruff lint, and Ruff formatting.
- ``make format-check`` — verify formatting without changing files.
- ``make check`` — type check.
- ``make test`` — quick tests.
- ``make test-slow`` — tests with real model examples.
