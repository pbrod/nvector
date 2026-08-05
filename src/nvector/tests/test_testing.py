from typing import Any

import pytest

import nvector
from nvector import testing

_PACKAGE_NAME = "nvector"


def test_test_function_exists() -> None:
    assert callable(nvector.test)


def test_test_calls_pytest_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def fake_main(args: list[str], plugins: list[Any] | None = None) -> int:
        called["args"] = args
        called["plugins"] = plugins
        return 0

    monkeypatch.setattr(pytest, "main", fake_main)

    rc = testing.test(_PACKAGE_NAME, "-q")

    assert rc == 0
    assert called["args"] == ["--pyargs", _PACKAGE_NAME, "-q"]
    assert called["plugins"] is None


def test_test_forwards_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def fake_main(args: list[str], plugins: list[Any] | None = None) -> int:
        called["args"] = args
        called["plugins"] = plugins
        return 0

    monkeypatch.setattr(pytest, "main", fake_main)

    plugin = object()

    testing.test(_PACKAGE_NAME, plugins=[plugin])

    assert called["plugins"] == [plugin]


def test_test_raises_helpful_error_without_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pytest":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(
        ImportError,
        match="pytest is required",
    ):
        testing.test(_PACKAGE_NAME)
