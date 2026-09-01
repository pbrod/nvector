import subprocess
import sys

import pytest

import nvector
from nvector import testing

_PACKAGE_NAME = "nvector"


def test_test_function_exists() -> None:
    assert callable(nvector.test)


def test_test_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, list[str]] = {}

    def fake_call(args: list[str]) -> int:
        called["args"] = args
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    rc = testing.test(_PACKAGE_NAME, "-q")

    assert rc == 0
    assert called["args"] == [
        sys.executable,
        "-m",
        "pytest",
        "--pyargs",
        _PACKAGE_NAME,
        "-q",
    ]
