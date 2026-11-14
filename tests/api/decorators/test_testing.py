"""Exercise skipping pytest tests that raise ``ImportError``."""

import asyncio

import pytest

from caresmcp.api.decorators.testing import skip_on_import_error


def test_returns_result_and_preserves_metadata():
    @skip_on_import_error
    def tool(value: int) -> int:
        return value * 2

    assert tool(21) == 42
    assert tool.__name__ == "tool"


@pytest.mark.parametrize(
    "error",
    [
        ImportError("missing optional package"),
        ModuleNotFoundError("No module named 'torch'"),
    ],
)
def test_import_error_skips(error: ImportError):
    @skip_on_import_error
    def tool():
        raise error

    with pytest.raises(pytest.skip.Exception) as excinfo:
        tool()
    assert str(excinfo.value) == str(error)


def test_other_exceptions_propagate():
    @skip_on_import_error
    def tool():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        tool()


def test_async_result_and_skip():
    @skip_on_import_error
    async def returns_value():
        return "ok"

    @skip_on_import_error
    async def raises_import_error():
        raise ImportError("missing optional package")

    assert asyncio.run(returns_value()) == "ok"
    with pytest.raises(pytest.skip.Exception, match="missing optional package"):
        asyncio.run(raises_import_error())
