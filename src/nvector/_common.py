import textwrap
import inspect


def _get_h1line(object_):
    """Returns the H1 line of the documentation of an object."""
    doc = object_.__doc__
    if doc:
        return doc.partition("Parameters\n")[0].strip()
    return ''


def _make_summary(odict):
    """Return summary of all functions and classes in odict"""
    prefix = '    '

    class_summary = '\n'.join([':\n'.join((oname, textwrap.indent(_get_h1line(obj), prefix)))
                               for oname, obj in odict.items() if inspect.isclass(obj)])

    fun_summary = '\n'.join([':\n'.join((oname, textwrap.indent(_get_h1line(obj), prefix)))
                             for oname, obj in odict.items() if not inspect.isclass(obj)])
    fmt = "{} in module\n{}----------\n{}\n\n"
    summary = ''
    if class_summary:
        summary = fmt.format("Classes", '-'*8, class_summary)
    if fun_summary:
        summary = summary + fmt.format('Functions', '-'*9, fun_summary)
    return summary


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
