from nvector import (_intro,
                     _documentation,
                     _installation,
                     _examples_object_oriented,
                     _acknowledgements,
                     _images)

__doc__ = ("""Introduction to nvector
=======================
""" + _intro.__doc__  # @UndefinedVariable @ReservedAssignment
           + _documentation.__doc__  # @UndefinedVariable
           + _installation.__doc__  # @UndefinedVariable
           + """Acknowledgements
================
""" + _acknowledgements.__doc__  # @UndefinedVariable
           + _examples_object_oriented.__doc__
           + """
See also
========
`geographiclib <https://pypi.python.org/pypi/geographiclib>`_


""" + _images.__doc__)  # @UndefinedVariable


if __name__ == '__main__':
    from nvector._common import write_readme, test_docstrings
    test_docstrings(__file__)
    write_readme(__doc__)
