from __future__ import annotations

from timeit import default_timer as timer
from typing import Any


def test_docstrings(filename: str) -> Any:
    import doctest

    if filename:
        print(f"Running doctests in {filename}...")
    else:
        print("Running doctests...")

    t0 = timer()
    result = doctest.testmod(optionflags=(doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS))
    dt = timer() - t0

    print(f"Attempted: {result.attempted}, Failed: {result.failed}, Elapsed: {dt:.3f}s")
    return result


def test(
    package_name: str,
    *options: str,
    plugins: Any | None = None,
) -> int:
    """
    Run tests for package using pytest.

    Parameters
    ----------
    package_name : str
        The name of the package to test.
    *options : optional
        options to pass to pytest. The most important ones include:
        '-v', '--verbose':
            increase verbosity.
        '-q', '--quiet':
            decrease verbosity.
        '--doctest-modules':
            run doctests in all .py modules
        '--cov':
            measure coverage for .py modules (requires pytest-cov plugin)
        '-h', '--help':
            show full help message and display all possible options to use.

    Returns
    -------
    exit_code: int
        Exit code is 0 if all tests passed without failure.

    Examples
    --------
    {super}

    """
    try:
        import pytest
    except ImportError as exc:
        raise ImportError(
            "pytest is required to run package tests. Install it with: pip install pytest."
        ) from exc

    return pytest.main(
        ["--pyargs", package_name, *options],
        plugins=plugins,
    )
