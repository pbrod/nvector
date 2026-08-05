import pytest

import nvector
from nvector import testing


def test_test_function_exists():
    assert callable(nvector.test)


def test_test_calls_pytest_main(monkeypatch):

    called = {}

    def fake_main(args, plugins=None):
        called["args"] = args
        called["plugins"] = plugins
        return 0

    import pytest

    monkeypatch.setattr(pytest, "main", fake_main)

    rc = testing.test("approxkit", "-q")

    assert rc == 0
    assert called["args"] == ["--pyargs", "approxkit", "-q"]
    assert called["plugins"] is None


def test_test_forwards_plugins(monkeypatch):

    called = {}

    def fake_main(args, plugins=None):
        called["plugins"] = plugins
        return 0


    monkeypatch.setattr(pytest, "main", fake_main)

    plugin = object()

    testing.test("approxkit", plugins=[plugin])

    assert called["plugins"] == [plugin]


def test_test_raises_helpful_error_without_pytest(monkeypatch):

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pytest":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(
        ImportError,
        match="pytest is required",
    ):
        testing.test("approxkit")