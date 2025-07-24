"""
Created on 26. sep. 2024

@author: pab
"""

import warnings
from typing import Any, Callable, TypeVar, Union

import numpy as np
import numpy.typing as npt

#ArrayLike = npt.ArrayLike  #
ArrayLike = Union[int, float, list, tuple, np.ndarray]
ArrayLikeTxt = "int, float, list, tuple or ndarray"

NpArrayLike = Union[np.float64, np.ndarray]
NpArrayLikeTxt = "float64 or ndarray"

#Array = npt.ArrayLike  #
Array = Union[list, tuple, np.ndarray]
ArrayTxt = "list, tuple or ndarray"

TYPES_DICT = {"array": ArrayTxt, "array_like": ArrayLikeTxt, "np_array_like": NpArrayLikeTxt}

F = TypeVar("F", bound=Callable[..., Any])


def format_docstring_types(func: F) -> F:
    """This decorator modifies the decorated function's docstring with supplied types.

    It replaces:
        "{array}" with ArrayTxt
        "{array_like}" with ArrayLikeTxt
        "{np_array_like}" with NpArrayLikeTxt

    This is useful when you want modify the docstring of a function at runtime.
    """

    func_docstring = func.__doc__
    if func_docstring is not None:
        try:
            new_docstring = func_docstring.format(**TYPES_DICT)
            func.__doc__ = new_docstring
        except Exception as error:
            warnings.warn(str(error), stacklevel=2)
            # python 2 crashes if the docstring already exists!
    return func
