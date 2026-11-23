# Contributing to CaresMCP

CaresMCP is research software, not a medical device. Contributions are welcome:
bug reports, feature and model proposals, documentation fixes, and code.

The [documentation](https://caresmcp.readthedocs.io/en/latest/contributing.html)
has the detailed guide for adding tools and updating model surveys. This file
covers the workflow shared by all changes.

## Before you start

- Search [existing issues](https://github.com/Moenupa/caresMCP/issues) and the
  [FAQ](https://caresmcp.readthedocs.io/en/latest/faq.html) first.
- Open an issue for non-trivial changes so the approach can be agreed before you
  write code. Small bug fixes and documentation fixes can go straight to a pull
  request.
- **Never include real patient data** in issues, tests, or examples. Use public
  or de-identified images, and remove paths and identifiers that could identify
  a patient.

## Development setup

Fork the repository, then clone your fork:

```console
git clone git@github.com:<your-handle>/caresMCP.git
cd caresMCP
git remote add upstream https://github.com/Moenupa/caresMCP.git
uv sync --all-extras
make
```

`make` runs the formatter, linter, and type checker. If you work on one
category, install only the extras you need and run the matching tests:

```console
uv sync --extra xray
uv run pytest tests/xray
```

Use a branch named `feat/xxx`, `fix/xxx`, or `doc/xxx`.

## Making a change

### Bug fixes

- Add a regression test under `tests/` that fails on `main` and passes with your
  fix. Include the test in the pull request.
- Keep the diff focused. Do not mix formatting, refactors, or unrelated fixes
  into the same pull request.
- If the bug is in an upstream model or package, say so in the issue. CaresMCP
  may only be able to document or work around it.

### Tools and models

Follow the [tool guide](https://caresmcp.readthedocs.io/en/latest/contributing.html#add-a-tool)
and the [model survey guide](https://caresmcp.readthedocs.io/en/latest/contributing.html#add-or-update-a-model-survey).
In short:

1. Add the function under `src/caresmcp/<category>/` and register it with the
   category and task registries.
2. Describe inputs with Pydantic `Field`; use the docstring for behavior, output
   interpretation, `References:`, and `See Also:`.
3. Add a test under `tests/<category>/` that exercises the public contract.
4. Run `make catalog` and commit the regenerated pages under
   `docs/capabilities/` and `docs/api/`.
5. For a model survey change, cite each model one-to-one and record its license
   and access terms.

### Documentation

Run `make docs` to build with warnings as errors. Generated API and catalog
pages are committed; edit the registry and regenerate them with `make catalog`
instead of editing the generated files.

## Checks before a pull request

Run the same checks as CI:

```console
make format-check
make check
make test
make catalog-check
```

`make catalog-check` is only relevant if you changed a registry entry, but it is
fast. Run `make test-slow` before a release or when a change touches model
loading; the default suite skips tests that download checkpoints.

## Pull requests

- Fill in the pull request template and link the issue with `Fixes #123` when one
  exists.
- If there is no issue, explain the problem and include a reproducer or
  before/after evidence in the pull request description.
- Keep one logical change per pull request. If you find an unrelated bug, open a
  separate issue or pull request.
- Wait for CI to pass before requesting review.
- Commit titles in this repository use `[area] summary`, for example
  `[xray] fix wrist reproduction`. This is not enforced, but it keeps the
  history readable.

## License

CaresMCP is licensed under the Apache License 2.0. By contributing, you agree
that your contributions are licensed under the same terms. See [LICENSE](LICENSE).
