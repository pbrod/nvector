""""""

from nvector import (
    _acknowledgements,
    _documentation,
    _examples_object_oriented,
    _images,
    _installation,
    _intro,
)

if __doc__ is not None:
    # Safely build module docstring even when frozen
    # (cx_Freeze) or run with -OO
    _sections = [
        "Introduction to nvector\n=======================\n",
        _intro.__doc__,
        _documentation.__doc__,
        _installation.__doc__,
        _examples_object_oriented.__doc__,
        "Acknowledgements\n================\n",
        _acknowledgements.__doc__,
        _images.__doc__,
    ]
    __doc__ = "".join(part for part in _sections if part)


if __name__ == "__main__":
    from nvector._common import write_readme
    from nvector.testing import test_docstrings

    write_readme(__doc__)
    test_docstrings(__file__)
