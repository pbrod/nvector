import pkg_resources
from ._info import __doc__  # @UnresolvedImport
from numpy.testing import Tester  # @UnresolvedImport
from ._core import *
from .objects import *

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    __version__ = 'unknown'

test = Tester(raise_warnings="release").test
