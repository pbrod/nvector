from typing import Any, Optional

from ._common import use_docstring
from ._info import __doc__ as __doc__  # @UnresolvedImport
from .core import *  # noqa
from .objects import *  # noqa
from .rotation import *  # noqa
from .util import *  # noqa

__version__ = "1.0.6"

_PACKAGE_NAME = __name__


@use_docstring(
    f"""
import {_PACKAGE_NAME} as {_PACKAGE_NAME[:2]}
{_PACKAGE_NAME[:2]}.test('-q', '--doctest-modules', '--cov={_PACKAGE_NAME}', '--disable-warnings')
"""
)
def test(*options: str, plugins: Optional[Any] = None) -> int:
    """
    Run tests for module using pytest.

    Parameters
    ----------
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
    import pytest

    return pytest.main(["--pyargs", _PACKAGE_NAME] + list(options), plugins=plugins)
