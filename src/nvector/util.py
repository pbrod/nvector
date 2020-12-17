"""
Utility functions
=================

"""
from __future__ import division, print_function
import warnings
from collections import namedtuple
import numpy as np
from numpy import rad2deg, deg2rad, deprecate
from numpy.linalg import norm
from nvector._common import test_docstrings, _make_summary
from nvector import license as _license

__all__ = ['deg', 'rad', 'mdot', 'nthroot', 'get_ellipsoid', 'select_ellipsoid', 'unit']


_EPS = np.finfo(float).eps  # machine precision (machine epsilon)
_TINY = np.finfo(float).tiny
Ellipsoid = namedtuple('Ellipsoid', 'a f name')

ELLIPSOID = {
    1: Ellipsoid(a=6377563.3960, f=1.0 / 299.3249646, name='Airy 1858'),
    2: Ellipsoid(a=6377340.189, f=1.0 / 299.3249646, name='Airy Modified'),
    3: Ellipsoid(a=6378160, f=1.0 / 298.25, name='Australian National'),
    4: Ellipsoid(a=6377397.155, f=1.0 / 299.1528128, name='Bessel 1841'),
    5: Ellipsoid(a=6378249.145, f=1.0 / 293.465, name='Clarke 1880'),
    6: Ellipsoid(a=6377276.345, f=1.0 / 300.8017, name='Everest 1830'),
    7: Ellipsoid(a=6377304.063, f=1.0 / 300.8017, name='Everest Modified'),
    8: Ellipsoid(a=6378166.0, f=1.0 / 298.3, name='Fisher 1960'),
    9: Ellipsoid(a=6378150.0, f=1.0 / 298.3, name='Fisher 1968'),
    10: Ellipsoid(a=6378270.0, f=1.0 / 297, name='Hough 1956'),
    11: Ellipsoid(a=6378388.0, f=1.0 / 297, name='International (Hayford)/European Datum (ED50)'),
    12: Ellipsoid(a=6378245.0, f=1.0 / 298.3, name='Krassovsky 1938'),
    13: Ellipsoid(a=6378145., f=1.0 / 298.25, name='NWL-9D  (WGS 66)'),
    14: Ellipsoid(a=6378160., f=1.0 / 298.25, name='South American 1969 (SAD69'),
    15: Ellipsoid(a=6378136, f=1.0 / 298.257, name='Soviet Geod. System 1985'),
    16: Ellipsoid(a=6378135., f=1.0 / 298.26, name='WGS 72'),
    17: Ellipsoid(a=6378206.4, f=1.0 / 294.9786982138, name='Clarke 1866    (NAD27)'),
    18: Ellipsoid(a=6378137.0, f=1.0 / 298.257223563, name='GRS80 / WGS84  (NAD83)'),
    19: Ellipsoid(a=6378137, f=298.257222101, name='ETRS89')
}
ELLIPSOID_IX = {'airy1858': 1,
                'airymodified': 2,
                'australiannational': 3,
                'bessel': 4,
                'bessel1841': 4,
                'clarke1880': 5,
                'everest1830': 6,
                'everestmodified': 7,
                'fisher1960': 8,
                'fisher1968': 9,
                'hough1956': 10,
                'hough': 10,
                'international': 11,
                'hayford': 11,
                'ed50': 11,
                'krassovsky': 12,
                'krassovsky1938': 12,
                'nwl-9d': 13,
                'wgs66': 13,
                'southamerican1969': 14,
                'sad69': 14,
                'sovietgeod.system1985': 15,
                'wgs72': 16,
                'clarke1866': 17,
                'nad27': 17,
                'grs80': 18,
                'wgs84': 18,
                'nad83': 18,
                }


def array_to_list_dict(data):
    """
    Convert dict arrays to dict of lists.

    Parameters
    ----------
    data : dict of arrays or an array

    Examples
    --------
    >>> import numpy as np
    >>> data = dict(a=np.zeros((3,)), b=(1,2,3), c=[], d=1, e='test',
    ...          f=np.nan, g=[1], h=[np.nan], i=None)
    >>> e = array_to_list_dict(data)
    >>> e == {'a': [0.0, 0.0, 0.0],  'b': [1, 2, 3], 'c': [],'d': 1,
    ...       'e': 'test', 'f': np.nan, 'g': [1], 'h': [np.nan], 'i': None}
    True

    """
    if isinstance(data, dict):
        for key in data:
            data[key] = array_to_list_dict(data[key])
    elif isinstance(data, (list, tuple)):
        data = [array_to_list_dict(item) for item in data]
    else:
        try:
            data = data.tolist()
        except AttributeError:
            pass
    return data


def isclose(a, b, rtol=1e-9, atol=0.0, equal_nan=False):
    """
    Returns True where the two arrays `a` and `b` are element-wise equal within a tolerance.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose

    Notes
    -----
    .. versionadded:: 0.7.5

    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent:

     absolute(`a` - `b`) <= maximimum(`atol`, `rtol` * maximum(absolute(`a`), absolute(`b`)))

    Like the built-in `math.isclose`, the above equation is symmetric
    in `a` and `b`. Furthermore, `atol` should be carefully selected for
    the use case at hand. A zero value for `atol` will result in `False`
    if either `a` or `b` is zero.

    Examples
    --------
    >>> import nvector.objects as no
    >>> no.isclose([1e10,1e-7], [1.00001e10,1e-8])
    array([False, False])
    >>> no.isclose([1e10,1e-8], [1.00001e10,1e-9])
    array([False, False])
    >>> no.isclose([1e10,1e-8], [1.0001e10,1e-9])
    array([False,  False])
    >>> no.isclose([1.0, np.nan], [1.0, np.nan])
    array([ True, False])
    >>> no.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    array([ True, True])
    >>> no.isclose([1e-8, 1e-7], [0.0, 0.0])
    array([False, False])
    >>> no.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
    array([False, False])
    >>> no.isclose([1e-10, 1e-10], [1e-20, 0.0])
    array([False,  False])
    >>> no.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
    array([False,  False])
    """
    a, b = np.broadcast_arrays(a, b)

    mask = np.isfinite(a) & np.isfinite(b)

    out = np.full(b.shape, False)
    abs_tol = np.maximum(atol, rtol*np.maximum(np.abs(a[mask]), np.abs(b[mask])))
    out[mask] = np.isclose(a[mask], b[mask], rtol=0, atol=abs_tol, equal_nan=equal_nan)
    mask = ~mask
    out[mask] = np.isclose(a[mask], b[mask], equal_nan=equal_nan)
    return out


def allclose(a, b, rtol=1.e-7, atol=1.e-14, equal_nan=False):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

        .. versionadded:: 1.10.0

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    isclose, all, any, equal

    Notes
    -----
    For finite values, allclose uses the following equation to test whether
    two floating point values are equivalent:

     absolute(`a` - `b`) <= maximimum(`atol`, `rtol` * maximum(absolute(`a`), absolute(`b`)))

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    The comparison of `a` and `b` uses standard broadcasting, which
    means that `a` and `b` need not have the same shape in order for
    ``allclose(a, b)`` to evaluate to True.

    Examples
    --------
    >>> import nvector.objects as no
    >>> no.allclose([1e10, 1e-7], [1.00001e10, 1e-8])
    False
    >>> no.allclose([1e10, 1e-8], [1.00001e10, 1e-9])
    False
    >>> no.allclose([1e10, 1e-8], [1.0001e10, 1e-9])
    False
    >>> no.allclose([1.0, np.nan], [1.0, np.nan])
    False
    >>> no.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    True

    """
    return np.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def _check_length_deviation(n_E, limit=0.1):
    """
    n-vector should have length=1,  i.e. norm(n_E)=1.

    A deviation from 1 exceeding this limit gives a warning.
    This function only depends of the direction of n-vector, thus the warning
    is included only to give a notice in cases where a wrong input is given
    unintentionally (i.e. the input is not even approximately a unit vector).

    If a matrix of n-vectors is input, only first is controlled to save time
    (assuming advanced users input correct n-vectors)
    """
    length_deviation = abs(norm(n_E[:, 0]) - 1)
    if length_deviation > limit:
        warnings.warn('n-vector should have unit length: '
                      'norm(n_E)~=1 ! Error is: {}'.format(length_deviation))


def deg(*rad_angles):
    """
    Converts angle in radians to degrees.

    Parameters
    ----------
    rad_angles:
        angle in radians

    Returns
    -------
    deg_angles:
        angle in degrees

    Examples
    --------
    >>> import numpy as np
    >>> import nvector as nv
    >>> nv.deg(np.pi/2)
    90.0
    >>> nv.deg(np.pi/2, [0, np.pi])
    (90.0, array([  0., 180.]))

    See also
    --------
    rad
    """
    if len(rad_angles) == 1:
        return rad2deg(rad_angles[0])
    return tuple(rad2deg(angle) for angle in rad_angles)


def rad(*deg_angles):
    """
    Converts angle in degrees to radians.

    Parameters
    ----------
    deg_angles:
        angle in degrees

    Returns
    -------
    rad_angles:
        angle in radians

    Examples
    --------
    >>> import numpy as np
    >>> import nvector as nv
    >>> nv.deg(nv.rad(90))
    90.0
    >>> nv.deg(*nv.rad(90, [0, 180]))
    (90.0, array([  0., 180.]))

    See also
    --------
    deg
    """
    if len(deg_angles) == 1:
        return deg2rad(deg_angles[0])
    return tuple(deg2rad(angle) for angle in deg_angles)


def mdot(a, b):
    """
    Returns multiple matrix multiplications of two arrays
    i.e.
      dot(a, b)[i,j,k] = sum(a[i,:,j] * b[:,j,k])

    if a and b have the same shape this is the same as:
      np.concatenate([np.dot(a[...,i], b[...,i])[:, :, None]
                      for i in range(n)], axis=2)

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Examples
    --------
    3 x 3 x 2 times 3 x 3 x 2 array -> 3 x 2 x 2 array
        >>> import numpy as np
        >>> import nvector as nv
        >>> a = 1.0 * np.arange(18).reshape(3,3,2)
        >>> b = - a
        >>> t = np.concatenate([np.dot(a[...,i], b[...,i])[:, :, None]
        ...                    for i in range(2)], axis=2)
        >>> tm = nv.mdot(a, b)
        >>> tm.shape
        (3, 3, 2)
        >>> np.allclose(t, tm)
        True

    3 x 3 x 2 times 3 x 1 array -> 3 x 1 x 2 array
        >>> t1 = np.concatenate([np.dot(a[...,i], b[:,0,0][:,None])[:,:,None]
        ...                    for i in range(2)], axis=2)

        >>> tm1 = nv.mdot(a, b[:,0,0].reshape(-1,1))
        >>> tm1.shape
        (3, 1, 2)
        >>> np.allclose(t1, tm1)
        True

    3 x 3  times 3 x 3 array -> 3 x 3 array
        >>> tt0 = nv.mdot(a[...,0], b[...,0])
        >>> tt0.shape
        (3, 3)
        >>> np.allclose(t[...,0], tt0)
        True

    3 x 3  times 3 x 1 array -> 3 x 1 array
        >>> tt0 = nv.mdot(a[...,0], b[:,:1,0])
        >>> tt0.shape
        (3, 1)
        >>> np.allclose(t[:,:1,0], tt0)
        True

    3 x 3  times 3 x 1 x 2 array -> 3 x 1 x 2 array
        >>> tt0 = nv.mdot(a[..., 0], b[:, :2, 0][:, None])
        >>> tt0.shape
        (3, 1, 2)
        >>> np.allclose(t[:,:2,0][:,None], tt0)
        True

    See also
    --------
    numpy.einsum
    """
    return np.einsum('ij...,jk...->ik...', a, b)


def nthroot(x, n):
    """
    Returns the n'th root of x to machine precision

    Parameters
    ----------
    x, n

    Examples
    --------
    >>> import numpy as np
    >>> import nvector as nv
    >>> np.allclose(nv.nthroot(27.0, 3), 3.0)
    True

    """
    y = x**(1. / n)
    return np.where((x != 0) & (_EPS * np.abs(x) < 1),
                    y - (y**n - x) / (n * y**(n - 1)), y)


def get_ellipsoid(name):
    """
    Returns semi-major axis (a), flattening (f) and name of ellipsoid as a named tuple.

    Parameters
    ----------
    name : string
        name of ellipsoid. Valid options are:
        1) Airy 1858
        2) Airy Modified
        3) Australian National
        4) Bessel 1841
        5) Clarke 1880
        6) Everest 1830
        7) Everest Modified
        8) Fisher 1960
        9) Fisher 1968
        10) Hough 1956
        11) International (Hayford)/European Datum (ED50)
        12) Krassovsky 1938
        13) NWL-9D (WGS 66)
        14) South American 1969
        15) Soviet Geod. System 1985
        16) WGS 72
        17) Clarke 1866    (NAD27)
        18) GRS80 / WGS84  (NAD83)
        19) ETRS89

    Examples
    --------
    >>> import nvector as nv
    >>> nv.get_ellipsoid(name='wgs84')
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84  (NAD83)')
    >>> nv.get_ellipsoid(name='GRS80')
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84  (NAD83)')
    >>> nv.get_ellipsoid(name='NAD83')
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84  (NAD83)')
    >>> nv.get_ellipsoid(name=18)
    Ellipsoid(a=6378137.0, f=0.0033528106647474805, name='GRS80 / WGS84  (NAD83)')

    >>> wgs72 = nv.select_ellipsoid(name="WGS 72")
    >>> wgs72.a == 6378135.0
    True
    >>> wgs72.f == 0.003352779454167505
    True
    >>> wgs72.name
    'WGS 72'
    >>> wgs72 == (6378135.0, 0.003352779454167505, 'WGS 72')
    True
    """
    if isinstance(name, str):
        name = name.lower().replace(' ', '')
    ellipsoid_id = ELLIPSOID_IX.get(name, name)

    return ELLIPSOID[ellipsoid_id]


select_ellipsoid = deprecate(get_ellipsoid, old_name='select_ellipsoid', new_name='get_ellipsoid')


def unit(vector, norm_zero_vector=1):
    """
    Convert input vector to a vector of unit length.

    Parameters
    ----------
    vector : 3 x m array
        m column vectors

    Returns
    -------
    unitvector : 3 x m array
        normalized unitvector(s) along axis==0.

    Notes
    -----
    The column vector(s) that have zero length will be returned as unit vector(s)
    pointing in the x-direction, i.e, [[1], [0], [0]]

    Examples
    --------
    >>> import numpy as np
    >>> import nvector as nv
    >>> np.allclose(nv.unit([[1, 0],[1, 0],[1, 0]]), [[ 0.57735027, 1],
    ...                                               [ 0.57735027, 0],
    ...                                               [ 0.57735027, 0]])
    True
    """
    current_norm = norm(vector, axis=0, keepdims=True)
    idx = np.flatnonzero(current_norm == 0)
    unit_vector = vector / (current_norm + _TINY)

    unit_vector[:, idx] = 0 * norm_zero_vector
    unit_vector[0, idx] = 1 * norm_zero_vector
    return unit_vector


_odict = globals()
__doc__ = (__doc__  # @ReservedAssignment
           + _make_summary(dict((n, _odict[n]) for n in __all__))
           + 'License\n-------\n'
           + _license.__doc__)


if __name__ == "__main__":
    test_docstrings(__file__)
