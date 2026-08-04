#!/usr/bin/env python
"""Update LICENSE.txt from src/nvector/_license.py."""

from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
LICENSE_MODULE = ROOT / "src" / "nvector" / "_license.py"
LICENSE_FILE = ROOT / "LICENSE.txt"


def load_docstring(module_path: Path) -> str:
    """Load __doc__ from a Python module."""
    spec = importlib.util.spec_from_file_location("_license", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.__doc__:
        raise ValueError(f"No module docstring found in {module_path}")

    return module.__doc__.strip() + "\n"


def main() -> None:
    text = load_docstring(LICENSE_MODULE)
    LICENSE_FILE.write_text(text, encoding="utf-8")
    print(f"Updated {LICENSE_FILE}")