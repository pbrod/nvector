from __future__ import absolute_import
from nvector._examples import getting_started
__doc__ = getting_started  # @ReservedAssignment


if __name__ == '__main__':
    from nvector._common import write_readme, test_docstrings
    test_docstrings(__file__)
    write_readme(__doc__)
