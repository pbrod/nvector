from functools import wraps
from typing import Any

from ._common import use_docstring
from ._info import __doc__ as __doc__  # @UnresolvedImport
from .core import *  # noqa
from .objects import *  # noqa
from .rotation import *  # noqa
from .testing import test as _test  # noqa
from .util import *  # noqa

__version__ = "1.0.6"

_PACKAGE_NAME = __name__


@use_docstring(
    f"""
import {_PACKAGE_NAME} as {_PACKAGE_NAME[:2]}
{_PACKAGE_NAME[:2]}.test('-q', '--doctest-modules', '--cov={_PACKAGE_NAME}', '--disable-warnings')
"""
)
@wraps(_test)
def test(*options: str, plugins: Any | None = None) -> int:
    return _test(__name__, *options, plugins=plugins)
