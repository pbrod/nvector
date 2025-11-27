"""
Created on 26. sep. 2024

@author: pab
"""

import warnings
from typing import Any, Callable, List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

# Using numpy.typing for better precision.
# ArrayLike is the most general type, accepting scalars, lists, tuples, and numpy arrays.
ArrayLike = npt.ArrayLike
ArrayLikeTxt = "npt.ArrayLike"

# NpArrayLike represents either a single numpy float or a numpy array of floats.
NpArrayLike = Union[np.floating, npt.NDArray[np.floating]]
NpArrayLikeTxt = "np.floating | npt.NDArray[np.floating]"


# IntArrayLike represents either a single numpy integer or a numpy array of integers.
IntArrayLike = Union[np.integer, npt.NDArray[np.integer]]
IntArrayLikeTxt = "np.integer | npt.NDArray[np.integer]"

NdArray = npt.NDArray[np.floating]
NdArrayTxt = "npt.NDArray[np.floating]"

BoolArray = npt.NDArray[np.bool_]
BoolArrayTxt = "npt.NDArray[np.bool_]"

# Array is a more specific type that can be a list, tuple, or numpy array.
Array = Union[List[Any], Tuple[Any, ...], npt.NDArray[Any]]
ArrayTxt = "List[Any] | Tuple[Any, ...] | npt.NDArray[Any]"

TYPES_DICT = {
    "array": ArrayTxt,
    "array_like": ArrayLikeTxt,
    "np_array_like": NpArrayLikeTxt,
    "int_array_like": IntArrayLikeTxt,
    "nd_array": NdArrayTxt,
    "bool_array": BoolArrayTxt,
}


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
