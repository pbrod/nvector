def use_docstring_from(cls):
    """
    This decorator modifies the decorated function's docstring by
    replacing it with the docstring from the class `cls`.
    """
    def _doc(func):
        cls_docstring = cls.__doc__
        func_docstring = func.__doc__
        if func_docstring is None:
            func.__doc__ = cls_docstring
        else:
            new_docstring = func_docstring % dict(super=cls_docstring)
            func.__doc__ = new_docstring
        return func
    return _doc


def test_docstrings(filename):
    import doctest
    print('Testing docstrings in {0!s}'.format(filename))
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    print('Docstrings tested')


def write_readme(doc):

    with open('readme.txt', 'w') as fid:
        fid.write(doc)
