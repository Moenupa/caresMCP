# AGENTS.md

## Commands

Use `make`, `uv`, and `uvx` commands to run code formatting, linting, and tests.

```sh
make                # lint, format, and type check
make format-check   # Check code lint/format (no fix)
make format         # lint and format
make check          # type check (may be skipped if too strict)
make test           # run test (lite)
make test-slow      # run test (full)
uv run pytest /path/to/test_file.py # Run specific test files
```

## Dependency Management

use `uv pip show <pkg>` to examine pkg dependencies. You should only write top-level pkgs.

## Code Style

Python 3.12, Google-style docstrings.
AssertionErrors are treated as internal errors: log as specific as possible.
Other exceptions are exposed to MCP server and may leak sensitive information.

## Framework

This is a medical tool library to be served as a single MCP server.

Our goal is to integrate validated medical tools into a unified framework, ensuring consistent handling of inputs and outputs while maintaining ease of use, maintainability, and extensibility.

We value:
- Stability and Access
- Modulization and delegation (e.g., transformers) instead of copying source code
- Zero-cost Inference (no runtime asserts, dev time `pytest` instead.)
