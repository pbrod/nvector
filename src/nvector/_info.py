from nvector import (
    _acknowledgements,
    _documentation,
    _examples_object_oriented,
    _images,
    _installation,
    _intro,
)

__doc__ = (  # @ReservedAssignment
    """Introduction to nvector
=======================
"""
    + _intro.__doc__  # @UndefinedVariable @ReservedAssignment
    + _documentation.__doc__  # @UndefinedVariable
    + _installation.__doc__  # @UndefinedVariable
    + _examples_object_oriented.__doc__
    + """Acknowledgements
================
"""
    + _acknowledgements.__doc__  # @UndefinedVariable
    + _images.__doc__
)  # @UndefinedVariable


if __name__ == "__main__":
    from nvector._common import test_docstrings, write_readme

    write_readme(__doc__)
    test_docstrings(__file__)
