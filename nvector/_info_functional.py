from nvector._examples import getting_started_functional
__doc__ = getting_started_functional  # @ReservedAssignment


def write_readme(doc):

    with open('readme.txt', 'w') as fid:
        fid.write(doc)


def test_docstrings():
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()

    # write_readme(__doc__)
