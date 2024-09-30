from ._info import __doc__  # @UnresolvedImport
from .util import *
from .rotation import *
from .core import *
from .objects import *
from ._common import use_docstring

__version__ = "0.7.7"

_PACKAGE_NAME = __name__


@use_docstring("""
import {0} as {1}
{1}.test('-q', '--doctest-modules', '--cov={0}', '--disable-warnings')
""".format(_PACKAGE_NAME, _PACKAGE_NAME[:2]))
def test(*options, plugins=None):
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
    exit_code: scalar
        Exit code is 0 if all tests passed without failure.

    Examples
    --------
    {super}

    """
    import pytest
    return pytest.main(['--pyargs', _PACKAGE_NAME] + list(options), plugins=plugins)
