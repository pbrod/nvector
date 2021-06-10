from nvector import (_intro,  # pylint: disable=no-name-in-module
                     _documentation,
                     _installation,
                     _examples_object_oriented,
                     _acknowledgements,
                     _images)

__doc__ = (  # @ReservedAssignment
    """Introduction to nvector
=======================
""" + _intro.__doc__  # @UndefinedVariable @ReservedAssignment
    + _documentation.__doc__  # @UndefinedVariable
    + _installation.__doc__  # @UndefinedVariable
    + _examples_object_oriented.__doc__
    + """Acknowledgements
================
""" + _acknowledgements.__doc__  # @UndefinedVariable
    + _images.__doc__)  # @UndefinedVariable


if __name__ == '__main__':
    from nvector._common import write_readme, test_docstrings
    test_docstrings(__file__)
    # write_readme(__doc__)
