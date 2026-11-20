.PHONY: all help format-check format check test test-slow list catalog catalog-check docs

check_dirs := src tests scripts

# uvx with fallback, e.g.: 1. `uvx ruff check` 2. `ruff check`
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")
SPHINXOPTS  ?= -T -W -b html -d docs/_build/doctrees -D language=en
DOCS_SRC    := docs
DOCS_OUT    := docs/_build/html

all: format check

help:
	@echo "Available targets:"
	@echo "  format-check:  Check code lint/format (no fix)"
	@echo "  format:        Run linter and formatter"
	@echo "  check:         Run type checker"
	@echo "  test:          Run quick tests"
	@echo "  test-slow:     Run tests with real examples (slow)"
	@echo "  catalog:       Regenerate the capability catalog pages"
	@echo "  catalog-check: Verify the generated catalog pages are current"
	@echo "  docs:          Build documentation in docs/_build/html"

format-check:
	$(TOOL) docstrfmt --check $(DOCS_SRC)
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

format:
	$(TOOL) docstrfmt $(DOCS_SRC)
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

check:
	$(TOOL) ty check $(check_dirs)

test:
	$(RUN) pytest tests -n 8

test-slow:
	RUN_SLOW=1 $(RUN) pytest tests -rA

list:
	$(RUN) caresmcp list

catalog:
	$(RUN) --no-dev scripts/generate_catalog.py

catalog-check:
	$(RUN) --no-dev scripts/generate_catalog.py --check

docs: catalog format
	rm -r $(DOCS_OUT)
	$(RUN) --extra docs sphinx-build $(SPHINXOPTS) $(DOCS_SRC) $(DOCS_OUT)
