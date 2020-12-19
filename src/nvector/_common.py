import inspect
try:
    import textwrap
    textwrap.indent  # pylint: disable=pointless-statement
except AttributeError:  # undefined function (wasn't added until Python 3.3)
    def indent(text, amount=4, ch=' '):
        padding = amount * ch
        return ''.join(padding+line for line in text.splitlines(True))
else:
    def indent(text, amount=4, ch=' '):
        return textwrap.indent(text, amount * ch)

dedent = textwrap.dedent


def _get_h1line(object_):
    """Returns the H1 line of the documentation of an object."""
    doc = object_.__doc__
    if doc:
        return doc.partition("Parameters\n")[0].strip()
    return ''


def _make_summary(odict):
    """Return summary of all functions and classes in odict"""

    class_summary = '\n'.join([':\n'.join((oname, indent(_get_h1line(obj))))
                               for oname, obj in odict.items() if inspect.isclass(obj)])

    fun_summary = '\n'.join([':\n'.join((oname, indent(_get_h1line(obj))))
                             for oname, obj in odict.items() if not inspect.isclass(obj)])
    fmt = "{} in module\n{}----------\n{}\n\n"
    summary = ''
    if class_summary:
        summary = fmt.format("Classes", '-'*8, class_summary)
    if fun_summary:
        summary = summary + fmt.format('Functions', '-'*9, fun_summary)
    return summary


def use_docstring_from(cls):
    """This decorator modifies the decorated function's docstring by
    with the docstring from the class `cls`.

    If the function's docstring is None it is replaced with the supplied cls.__doc__.
    otherwise it is set to old_docstring.format(super=cls.__doc__)

    This is useful when you want to reuse the docstring from another class or
    if you want modify the docstring of a function at runtime.

    """
    return use_docstring(cls.__doc__)


def use_docstring(docstring):
    """This decorator modifies the decorated function's docstring with supplied docstring.

    If the function's docstring is None it is replaced with the supplied docstring.
    otherwise it is set to old_docstring.format(super=docstring)

    This is useful when you want modify the docstring of a function at runtime.
    """
    def _doc(func):
        func_docstring = func.__doc__
        if func_docstring is None:
            func.__doc__ = docstring
        else:
            try:
                new_docstring = dedent(func_docstring).format(super=docstring)
                func.__doc__ = new_docstring
            except Exception:
                pass  # python 2 crashes if the docstring alreasy exists!
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
