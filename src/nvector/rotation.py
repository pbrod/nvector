"""
Module related to rotation matrices and angles
==============================================

"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy import arctan2, sin, cos, array, sqrt, ndarray
from nvector import _license
from nvector._common import test_docstrings, _make_summary
from nvector.util import mdot, unit, norm, _nvector_check_length
from nvector._typing import ArrayLike, NpArrayLike, Array, format_docstring_types

__all__ = [
    "E_rotation",
    "n_E_and_wa2R_EL",
    "n_E2R_EN",
    "R_EL2n_E",
    "R_EN2n_E",
    "R2xyz",
    "R2zyx",
    "xyz2R",
    "zyx2R",
    "change_axes_to_E",
]

_EPS = np.finfo(float).eps
E_ROTATION_MATRIX = dict(e=np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]), E=np.eye(3))
"""Rotation matrix defining the axes of the coordinate frame E."""

# pylint: disable=invalid-name


def E_rotation(axes: str = "e") -> ndarray:
    """
    Returns rotation matrix R_Ee defining the axes of the coordinate frame E.

    Parameters
    ----------
    axes : str
        Either "e" or "E" defining the orientation of the axes of the coordinate frame E.
        If axes is "e" then z-axis points to the North Pole along the Earth's
        rotation axis, x-axis points towards the point where latitude = longitude = 0.
        If axes is "E" then x-axis points to the North Pole along the Earth's
        rotation axis, y-axis points towards where longitude +90deg (east) and latitude = 0.

    Returns
    -------
    R_Ee : ndarray
        3 x 3 rotation matrix defining the axes of the coordinate frame E as
        described in Table 2 in the article by Gade :cite:`Gade2010Nonsingular`.

    Notes
    -----
    R_Ee controls the axes of the coordinate frame E (Earth-Centred,
    Earth-Fixed, ECEF) used by the other functions in this library.
    It is very common in many fields to choose axes equal to "e", which
    is also the default in this library. Previously the old matlab toolbox
    the default value was equal to "E".
    If you choose axes equal to "E" the yz-plane coincides with the equatorial
    plane. This choice of axis ensures that at zero latitude and longitude,
    frame N (North-East-Down) has the same orientation as frame E. If
    roll/pitch/yaw are zero, also frame B (forward-starboard-down) has this
    orientation. In this manner, the axes of frame E is chosen to correspond
    with the axes of frame N and B.

    Examples
    --------
    >>> import numpy as np
    >>> import nvector as nv
    >>> bool(np.allclose(nv.E_rotation(axes="e"), [[ 0,  0,  1],
    ...                                            [ 0,  1,  0],
    ...                                            [-1,  0,  0]]))
    True
    >>> bool(np.allclose(nv.E_rotation(axes="E"), [[ 1.,  0.,  0.],
    ...                                            [ 0.,  1.,  0.],
    ...                                            [ 0.,  0.,  1.]]))
    True

    """
    return E_ROTATION_MATRIX[axes]


@format_docstring_types
def R2xyz(R_AB: Array) -> tuple[NpArrayLike, NpArrayLike, NpArrayLike]:
    """
    Returns the Euler angles in the xyz-order from a rotation matrix.

    Parameters
    ----------
    R_AB : {array}
        3 x 3 x n rotation array [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = mdot(R_AB, v_B).

    Returns
    -------
    x, y, z: {np_array_like}
        Angles [rad] of rotation about new axes given as real scalars or vectors of length n.

    Notes
    -----
    The x, y, z angles are called Euler angles or Tait-Bryan angles and are
    defined by the following procedure of successive rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle x about its x-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle z about its NEWEST z-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    See also:
    https://en.wikipedia.org/wiki/Aircraft_principal_axes
    https://en.wikipedia.org/wiki/Euler_angles
    https://en.wikipedia.org/wiki/Axes_conventions

    See also
    --------
    xyz2R, R2zyx, xyz2R
    """
    R_AB = np.atleast_2d(R_AB)

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = sqrt(
        (R_AB[0, 0, ...] ** 2 + R_AB[0, 1, ...] ** 2 + R_AB[1, 2, ...] ** 2 + R_AB[2, 2, ...] ** 2)
        / 2
    )
    sin_y = R_AB[0, 2, ...]

    non_singular = cos_y > 10 * _EPS  # atan2: [-pi pi]
    x = np.where(
        non_singular, arctan2(-R_AB[1, 2, ...], R_AB[2, 2, ...]), 0
    )  # Only the sum/difference of x and z is now given, choosing x = 0
    y = np.where(
        non_singular, arctan2(sin_y, cos_y), np.sign(sin_y) * np.pi / 2
    )  # Selecting y = +-pi/2, with correct sign
    z = np.where(
        non_singular,
        arctan2(-R_AB[0, 1, ...], R_AB[0, 0, ...]),
        # Lower left 2x2 elements of R_AB now only consists of sin_z and cos_z.
        # Using the two whose signs are the same for both singularities:
        arctan2(R_AB[1, 0, ...], R_AB[1, 1, ...]),
    )
    return x, y, z


@format_docstring_types
def R2zyx(R_AB: Array) -> tuple[NpArrayLike, NpArrayLike, NpArrayLike]:
    """
    Returns the Euler angles in the zxy-order from a rotation matrix.

    Parameters
    ----------
    R_AB : {array}
        3 x 3 x n rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = mdot(R_AB, v_B).

    Returns
    -------
    z, y, x: {np_array_like}
        Angles [rad] of rotation about new axes given as real scalars or vectors of length n.

    Notes
    -----
    The z, x, y angles are called Euler angles or Tait-Bryan angles and are
    defined by the following procedure of successive rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle z about its z-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle x about its NEWEST x-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    Note that if A is a north-east-down frame and B is a body frame, we
    have that z=yaw, y=pitch and x=roll.

    See also:
    https://en.wikipedia.org/wiki/Aircraft_principal_axes
    https://en.wikipedia.org/wiki/Euler_angles
    https://en.wikipedia.org/wiki/Axes_conventions

    See also
    --------
    zyx2R, xyz2R, R2xyz
    """
    x, y, z = R2xyz(np.swapaxes(R_AB, 1, 0))
    return -z, -y, -x


@format_docstring_types
def R_EL2n_E(R_EL: Array) -> ndarray:
    """
    Returns n-vector from the rotation matrix R_EL.

    Parameters
    ----------
    R_EL: {array}
        3 x 3 x n rotation matrix (direction cosine matrix) [no unit].

    Returns
    -------
    n_E: ndarray
        3 x n array of n-vector(s) [no unit] decomposed in E.

    Notes
    -----
    n-vector is found from the rotation matrix (direction cosine matrix) R_EL.

    See also
    --------
    R_EN2n_E, n_E_and_wa2R_EL, n_E2R_EN
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = mdot(R_EL, np.vstack((0, 0, -1)))
    return n_E.reshape(3, -1)


@format_docstring_types
def R_EN2n_E(R_EN: Array) -> ndarray:
    """
    Returns n-vector from the rotation matrix R_EN.

    Parameters
    ----------
    R_EN: {array}
        3 x 3 x n rotation matrix (direction cosine matrix) [no unit].

    Returns
    -------
    n_E: ndarray
        3 x n array of n-vector [no unit] decomposed in E.

    Notes
    -----
    n-vector is found from the rotation matrix (direction cosine matrix) R_EN.

    See also
    --------
    n_E2R_EN, R_EL2n_E, n_E_and_wa2R_EL
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    return R_EL2n_E(R_EN)


def _atleast_3d(x, y, z):
    """
    Examples
    --------
    >>> from nvector.rotation import _atleast_3d
    >>> for arr in _atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
    ...     print(arr, arr.shape)
    [[[[[1 2]]]]] (1, 1, 1, 1, 2)
    [[[[[1 2]]]]] (1, 1, 1, 1, 2)
    [[[[[1 2]]]]] (1, 1, 1, 1, 2)

    >>> for arr in _atleast_3d([[1], [2]], [[[1], [2]]], [[[[1], [2]]]]):
    ...     print(arr, arr.shape)
    [[[[[[1]
         [2]]]]]] (1, 1, 1, 1, 2, 1)
    [[[[[[1]
         [2]]]]]] (1, 1, 1, 1, 2, 1)
    [[[[[[1]
         [2]]]]]] (1, 1, 1, 1, 2, 1)
    """
    x, y, z = np.broadcast_arrays(*np.atleast_1d(x, y, z))
    return x[None, None, :], y[None, None, :], z[None, None, :]


@format_docstring_types
def xyz2R(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ndarray:
    """
    Returns rotation matrix from Euler angles in the xyz-order.

    Parameters
    ----------
    x, y, z: {array_like}
        Angles [rad] of rotation about new axes given as real scalars or array of lengths n.

    Returns
    -------
    R_AB: ndarray
        3 x 3 x n rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = mdot(R_AB, v_B).

    Notes
    -----
    The rotation matrix R_AB is created based on 3 angles x,y,z about new axes
    (intrinsic) in the order x-y-z. The angles are called Euler angles or
    Tait-Bryan angles and are defined by the following procedure of successive
    rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle x about its x-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle z about its NEWEST z-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    See also:
    https://en.wikipedia.org/wiki/Aircraft_principal_axes
    https://en.wikipedia.org/wiki/Euler_angles
    https://en.wikipedia.org/wiki/Axes_conventions

    See also
    --------
    R2xyz, zyx2R, R2zyx
    """
    x, y, z = _atleast_3d(x, y, z)
    sx, sy, sz = sin(x), sin(y), sin(z)
    cx, cy, cz = cos(x), cos(y), cos(z)

    R_AB = array(
        [
            [cy * cz, -cy * sz, sy],
            [sy * sx * cz + cx * sz, -sy * sx * sz + cx * cz, -cy * sx],
            [-sy * cx * cz + sx * sz, sy * cx * sz + sx * cz, cy * cx],
        ]
    )

    return np.squeeze(R_AB)


@format_docstring_types
def zyx2R(z: ArrayLike, y: ArrayLike, x: ArrayLike) -> ndarray:
    """
    Returns rotation matrix from Euler angles in the zyx-order.

    Parameters
    ----------
    z, y, x: {array_like}
        Angles [rad] of rotation about new axes given as real scalars or arrays of lenths n.

    Returns
    -------
    R_AB: ndarray
        3 x 3 x n rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = mdot(R_AB, v_B).

    Notes
    -----
    The rotation matrix R_AB is created based on 3 angles
    z,y,x about new axes (intrinsic) in the order z-y-x. The angles are called
    Euler angles or Tait-Bryan angles and are defined by the following
    procedure of successive rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle z about its z-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle x about its NEWEST x-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    Note that if A is a north-east-down frame and B is a body frame, we
    have that z=yaw, y=pitch and x=roll.

    See also:
    https://en.wikipedia.org/wiki/Aircraft_principal_axes
    https://en.wikipedia.org/wiki/Euler_angles
    https://en.wikipedia.org/wiki/Axes_conventions

    Examples
    --------
    Suppose the yaw angle between coordinate system A and B is 45 degrees.
    Convert position p1_b = (1, 0, 0) in B to a point in A.
    Convert position p2_a =(0, 1, 0) in A to a point in B.

    Solution:
        >>> import numpy as np
        >>> import nvector as nv
        >>> x, y, z = nv.rad(0, 0, 45)
        >>> R_AB = nv.zyx2R(z, y, x)

        >>> p1_b = np.atleast_2d((1, 0, 0)).T
        >>> p1_a = nv.mdot(R_AB, p1_b)
        >>> bool(nv.allclose(p1_a, [[0.7071067811865476], [0.7071067811865476], [0.0]]))
        True

        >>> p2_a = np.atleast_2d((0, 1, 0)).T
        >>> p2_b = nv.mdot(R_AB.T, p2_a)
        >>> bool(nv.allclose(p2_b, [[0.7071067811865476], [0.7071067811865476], [0.0]]))
        True

    See also
    --------
    R2zyx, xyz2R, R2xyz
    """
    x, y, z = _atleast_3d(x, y, z)
    sx, sy, sz = sin(x), sin(y), sin(z)
    cx, cy, cz = cos(x), cos(y), cos(z)

    R_AB = array(
        [
            [cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy * cx],
            [sz * cy, cz * cx + sz * sy * sx, -cz * sx + sz * sy * cx],
            [-sy, cy * sx, cy * cx],
        ]
    )

    return np.squeeze(R_AB)


@format_docstring_types
def n_E2lat_lon(n_E: Array, R_Ee: Optional[Array] = None) -> tuple[ndarray, ndarray]:
    """
    Converts n-vector(s) to latitude(s) and longitude(s).

    Parameters
    ----------
    n_E: {array}
        3 x n array of n-vector(s) [no unit] decomposed in E.
    R_Ee : {array}
        3 x 3  rotation matrix defining the axes of the coordinate frame E,
        default E_rotation().

    Returns
    -------
    latitude, longitude: ndarray
        Geodetic latitude(s) and longitude(s) given in [rad]

    See also
    --------
    lat_lon2n_E, nvector.rotation.E_rotation
    """

    n_e = change_axes_to_E(n_E, R_Ee)

    sin_latitude = n_e[0, ...]
    cos_latitude = sqrt(n_e[1, ...] ** 2 + n_e[2, ...] ** 2)
    sin_longitude_cos_latitude = n_e[1, ...]
    cos_longitude_cos_latitude = -n_e[2, ...]

    # Equation (5) in Gade (2010):
    longitude = arctan2(sin_longitude_cos_latitude, cos_longitude_cos_latitude)
    # Equation (6) in Gade (2010) (Robust numerical solution)
    latitude = arctan2(sin_latitude, cos_latitude)
    # atan() could also be used since latitude is within [-pi/2,pi/2]

    # latitude=asin(n_e[0] is a theoretical solution, but close to the Poles
    # it is ill-conditioned which may lead to numerical inaccuracies (and it
    # will give imaginary results for norm(n_E)>1)
    return latitude, longitude


@format_docstring_types
def change_axes_to_E(n_E: Array, R_Ee: Optional[Array] = None) -> ndarray:
    """
    Change axes of the nvector(s) from "e" to "E".

    Parameters
    ----------
    n_E: {array}
        3 x n array of n-vector(s) [no unit] decomposed in E.
    R_Ee : {array}
        3 x 3 rotation matrix defining the axes of the coordinate frame E,
        default E_rotation().

    Returns
    -------
    n_e: ndarray
        3 x n array of n-vector(s) [no unit] decomposed in e.

    Notes
    -----
    The function make sure to rotate the coordinates so that axes is "E":
    then x-axis points to the North Pole along the Earth's rotation axis,
    and yz-plane coincides with the equatorial plane, i.e.,
    y-axis points towards longitude +90deg (east) and latitude = 0.

    See also
    --------
    E_rotation
    """
    if R_Ee is None:
        R_Ee = E_rotation()

    n_E = np.atleast_2d(n_E)
    _nvector_check_length(n_E)

    n_e = unit(np.matmul(R_Ee, n_E))
    return n_e


@format_docstring_types
def n_E2R_EN(n_E: Array, R_Ee: Optional[Array] = None) -> ndarray:
    """
    Returns the rotation matrix R_EN from n-vector.

    Parameters
    ----------
    n_E: {array}
        3 x n array of n-vector(s) [no unit] decomposed in E
    R_Ee : {array}
        3 x 3 rotation matrix defining the axes of the coordinate frame E,
        default E_rotation().

    Returns
    -------
    R_EN:  ndarray
        The resulting 3 x 3 x n rotation matrix [no unit] (direction cosine matrix).

    See also
    --------
    R_EN2n_E, n_E_and_wa2R_EL, R_EL2n_E, E_rotation
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    #     n_E = np.atleast_2d(n_E)
    #     _nvector_check_length(n_E)
    #     n_E = unit(np.matmul(R_Ee, n_E))
    n_e = change_axes_to_E(n_E, R_Ee)

    # N coordinate frame (North-East-Down) is defined in Table 2 in Gade (2010)
    # Find z-axis of N (Nz):
    Nz_e = -n_e  # z-axis of N (down) points opposite to n-vector

    # Find y-axis of N (East)(remember that N is singular at Poles)
    # Equation (9) in Gade (2010):
    # Ny points perpendicular to the plane
    Ny_e_direction = np.cross([[1], [0], [0]], n_e, axis=0)
    # formed by n-vector and Earth's spin axis
    on_poles = np.flatnonzero(norm(Ny_e_direction, axis=0) == 0)
    Ny_e = unit(Ny_e_direction)
    Ny_e[:, on_poles] = array([[0], [1], [0]])  # selected y-axis direction

    # Find x-axis of N (North):
    Nx_e = np.cross(Ny_e, Nz_e, axis=0)  # Final axis found by right hand rule

    # Form R_EN from the unit vectors:
    # R_EN = dot(R_Ee.T, np.hstack((Nx_e, Ny_e, Nz_e)))
    Nxyz_e = np.hstack((Nx_e[:, None, ...], Ny_e[:, None, ...], Nz_e[:, None, ...]))
    R_EN = mdot(np.swapaxes(R_Ee, 1, 0), Nxyz_e)

    return np.squeeze(R_EN)


@format_docstring_types
def n_E_and_wa2R_EL(n_E: Array, wander_azimuth: ArrayLike, R_Ee: Optional[Array] = None) -> ndarray:
    """
    Returns rotation matrix R_EL from n-vector and wander azimuth angle.

    Parameters
    ----------
    n_E : {array}
        3 x n array of n-vector(s) [no unit] decomposed in E.
    wander_azimuth: {array_like}
        Angle(s) [rad] between L's x-axis and north, positive about L's z-axis given
        as a real scalar or array of length n.
    R_Ee : {array}
        3 x 3 rotation matrix defining the axes of the coordinate frame E,
        default E_rotation().

    Returns
    -------
    R_EL: ndarray
        The resulting 3 x 3 x n rotation matrix.  [no unit]

    Notes
    -----
    Calculates the rotation matrix (direction cosine matrix) `R_EL` using
    n-vector (`n_E`) and the wander azimuth angle. When `wander_azimuth`=0, we
    have that N=L. (See Table 2 in Gade :cite:`Gade2010Nonsingular` for details)

    See also
    --------
    R_EL2n_E,
    R_EN2n_E,
    n_E2R_EN,
    nvector.rotation.E_rotation
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    latitude, longitude = n_E2lat_lon(n_E, R_Ee)

    # Longitude, -latitude, and wander azimuth are the x-y-z Euler angles (about
    # new axes) for R_EL.
    # Reference: See start of Section 5.2 in Gade (2010):
    R_EL = mdot(R_Ee.T, xyz2R(longitude, -latitude, wander_azimuth))
    return np.squeeze(R_EL)


_odict = globals()
__doc__ = (
    __doc__  # @ReservedAssignment
    + _make_summary(dict((n, _odict[n]) for n in __all__))
    + '.. only:: draft\n\n'
    + "    License\n    -------\n    "
    + _license.__doc__.replace('\n', '\n    ')
)


if __name__ == "__main__":
    test_docstrings(__file__)
