---
name: write-tests
description: >-
  Write or update tests for tools, APIs or helper functions.
---

# Write tests

The sole purpose of tests is to check the correctness of our implementation.

Prefer end-to-end, tool-level tests with real inputs: reproducing a claimed
result beats checking shapes. Test `api/` helpers shared by several tools, but
skip private helpers local to one tool, the MCP server, and tests that are
obvious or could not pass type checking. Never fake or synthesize inputs to make
a test pass.

If public score reproduction is not possible, fallback to check the code.
Avoid self-reported scores.

## Layout

```text
tests/<category>/
  conftest.py          # session-scoped fixtures with real URLs
  test_<module>.py     # one file per tool module
```

Fixtures read real inputs through the same byte reader used by tools:

```python
import pytest
from caresmcp.api.decorators.bytes_io import _read_bytes

@pytest.fixture(scope="session")
def pathology_image() -> bytes:
    # reduce github urls, as they can be rate-limited
    return _read_bytes("https://example.org/real-slide-region.png")
```

Avoid full dataset downloading or checking. Our tests should be minimal,
but tests the core functionalities.

## Test tools

```python
import pytest
from caresmcp.api.decorators import save_returns

pytestmark = pytest.mark.slow

@save_returns("xray_classify")
def test_classify(hernia_image: bytes, model_id: str):
    result = target_classify_tool(hernia_image, model_id=model_id)
    # Cheap contract checks first.
    assert result["model_id"] == model_id
    # Then quality: assert it classifies as expected, if the model supports it.
    assert result["scores"].get("Hernia", 1.0) > 0.5
    # When reproduction is feasible, pin exact values within relative 1%.
    # assert result["scores"]["Hernia"] == pytest.approx(0.99, rel=1e-2)
    return result
```

- Omit return annotations on tests. Return the result so `save_returns` persists
  it for inspection; a non-JSON-serializable, non-`bytes` result fails there.
- Parametrize rather than duplicating tests, e.g. across `Literal` model ids
  with `typing.get_args`.

## Markers and decorators

- `@pytest.mark.slow`: real inputs or weight downloads; skipped unless
  `RUN_SLOW=1`.
- `@pytest.mark.cuda`: large (>100M parameters) or CUDA-only models; skipped
  unless `caresmcp.constants.DEVICE == "cuda"`.
- `@save_returns("collection")`: save under `CACHE_ROOT/collection/` with a
  `.result.json` or detected binary `.result.<ext>` suffix. With `unpack=True`,
  expect a dict of named artifacts and save each under `<test-node>/`.
- `@skip_on_import_error`: report a missing optional dependency as a skip.
- `@skip_on_exception(ImportError, OSError)`: also skip gated or unreachable
  Hugging Face downloads. Use only for environmental failures, not real errors.

## Running

```sh
uv run pytest /path/to/test_file.py
make test        # all fast tests, pytest -n 8
make test-slow   # RUN_SLOW=1, includes real downloads
```

The root `tests/conftest.py` performs the `slow`/`cuda` skipping; do not
reimplement it in category conftests.
