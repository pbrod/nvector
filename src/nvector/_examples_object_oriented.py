from nvector._examples import GETTING_STARTED
__doc__ = GETTING_STARTED  # @ReservedAssignment


if __name__ == '__main__':
    from nvector._common import write_readme, test_docstrings
    test_docstrings(__file__)
    write_readme(__doc__)
