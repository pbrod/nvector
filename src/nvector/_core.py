"""
This file is part of NavLab and is available from www.navlab.net/nvector

The content of this file is based on the following publication:

Gade, K. (2010). A Nonsingular Horizontal Position Representation, The Journal
of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
(www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf)

This paper should be cited in publications using this file.

Copyright (c) 2015, Norwegian Defence Research Establishment (FFI)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above publication
information, copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above publication
information, copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import division, print_function
import warnings
import numpy as np
from numpy import rad2deg, deg2rad, arctan2, sin, cos, array, cross, dot, sqrt
from numpy.linalg import norm
from nvector import _examples
from nvector._common import test_docstrings, use_docstring_from


__all__ = ['select_ellipsoid', 'E_rotation',
           'unit', 'deg', 'rad', 'nthroot',
           'lat_lon2n_E', 'n_E2lat_lon',
           'n_EA_E_and_n_EB_E2p_AB_E', 'n_EA_E_and_p_AB_E2n_EB_E',
           'p_EB_E2n_EB_E', 'n_EB_E2p_EB_E',
           'n_EA_E_distance_and_azimuth2n_EB_E',
           'n_EA_E_and_n_EB_E2azimuth',
           'great_circle_distance', 'euclidean_distance',
           'great_circle_normal', 'cross_track_distance',
           'closest_point_on_great_circle', 'on_great_circle',
           'on_great_circle_path', 'interpolate',
           'intersect', 'mean_horizontal_position',
           'R2xyz', 'xyz2R', 'R2zyx', 'zyx2R', 'mdot',
           'n_E_and_wa2R_EL', 'n_E2R_EN', 'R_EL2n_E', 'R_EN2n_E']

E_ROTATION_MATRIX = dict(e=array([[0, 0, 1.0],
                                  [0, 1.0, 0],
                                  [-1.0, 0, 0]]),
                         E=np.eye(3))

_EPS = np.finfo(float).eps  # machine precision (machine epsilon)
_TINY = np.finfo(float).tiny


ELLIPSOID = {1: ({'a': 6377563.3960, 'f': 1.0 / 299.3249646}, 'Airy 1858'),
             2: ({'a': 6377340.189, 'f': 1.0 / 299.3249646}, 'Airy Modified'),
             3: ({'a': 6378160, 'f': 1.0 / 298.25}, 'Australian National'),
             4: ({'a': 6377397.155, 'f': 1.0 / 299.1528128}, 'Bessel 1841'),
             5: ({'a': 6378249.145, 'f': 1.0 / 293.465}, 'Clarke 1880'),
             6: ({'a': 6377276.345, 'f': 1.0 / 300.8017}, 'Everest 1830'),
             7: ({'a': 6377304.063, 'f': 1.0 / 300.8017}, 'Everest Modified'),
             8: ({'a': 6378166.0, 'f': 1.0 / 298.3}, 'Fisher 1960'),
             9: ({'a': 6378150.0, 'f': 1.0 / 298.3}, 'Fisher 1968'),
             10: ({'a': 6378270.0, 'f': 1.0 / 297}, 'Hough 1956'),
             11: ({'a': 6378388.0, 'f': 1.0 / 297},
                  'International (Hayford)/European Datum (ED50)'),
             12: ({'a': 6378245.0, 'f': 1.0 / 298.3}, 'Krassovsky 1938'),
             13: ({'a': 6378145., 'f': 1.0 / 298.25}, 'NWL-9D  (WGS 66)'),
             14: ({'a': 6378160., 'f': 1.0 / 298.25}, 'South American 1969 (SAD69'),
             15: ({'a': 6378136, 'f': 1.0 / 298.257},
                  'Soviet Geod. System 1985'),
             16: ({'a': 6378135., 'f': 1.0 / 298.26}, 'WGS 72'),
             17: ({'a': 6378206.4, 'f': 1.0 / 294.9786982138},
                  'Clarke 1866    (NAD27)'),
             18: ({'a': 6378137.0, 'f': 1.0 / 298.257223563},
                  'GRS80 / WGS84  (NAD83)'),
             19: ({'a': 6378137, 'f': 298.257222101}, 'ETRS89')
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


def select_ellipsoid(name):
    """
    Returns semi-major axis (a), flattening (f) and name of ellipsoid

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
    >>> nv.select_ellipsoid(name='wgs84')
    (6378137.0, 0.0033528106647474805, 'GRS80 / WGS84  (NAD83)')
    >>> nv.select_ellipsoid(name='GRS80')
    (6378137.0, 0.0033528106647474805, 'GRS80 / WGS84  (NAD83)')
    >>> nv.select_ellipsoid(name='NAD83')
    (6378137.0, 0.0033528106647474805, 'GRS80 / WGS84  (NAD83)')
    >>> nv.select_ellipsoid(name=18)
    (6378137.0, 0.0033528106647474805, 'GRS80 / WGS84  (NAD83)')
    """
    if isinstance(name, str):
        name = name.lower().replace(' ', '')
    ellipsoid_id = ELLIPSOID_IX.get(name, name)

    ellipsoid, fullname = ELLIPSOID[ellipsoid_id]
    return ellipsoid['a'], ellipsoid['f'], fullname


def E_rotation(axes='e'):
    """
    Returns rotation matrix R_Ee defining the axes of the coordinate frame E.

    Parameters
    ----------
    axes : 'e' or 'E'
        defines orientation of the axes of the coordinate frame E. Options are:
        'e': z-axis points to the North Pole along the Earth's rotation axis,
             x-axis points towards the point where latitude = longitude = 0.
             This choice is very common in many fields.
        'E': x-axis points to the North Pole along the Earth's rotation axis,
             y-axis points towards longitude +90deg (east) and latitude = 0.
             (the yz-plane coincides with the equatorial plane).
             This choice of axis ensures that at zero latitude and longitude,
             frame N (North-East-Down) has the same orientation as frame E.
             If roll/pitch/yaw are zero, also frame B (forward-starboard-down)
             has this orientation. In this manner, the axes of frame E is
             chosen to correspond with the axes of frame N and B.
             The functions in this library originally used this option.

    Returns
    -------
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E as
        described in Table 2 in Gade (2010)

    R_Ee controls the axes of the coordinate frame E (Earth-Centred,
    Earth-Fixed, ECEF) used by the other functions in this library

    Examples
    --------
    >>> import numpy as np
    >>> import nvector as nv
    >>> np.allclose(nv.E_rotation(axes='e'), [[ 0,  0,  1],
    ...                                       [ 0,  1,  0],
    ...                                       [-1,  0,  0]])
    True
    >>> np.allclose(nv.E_rotation(axes='E'), [[ 1.,  0.,  0.],
    ...                                       [ 0.,  1.,  0.],
    ...                                       [ 0.,  0.,  1.]])
    True

    Reference
    ---------
    Gade, K. (2010). `A Nonsingular Horizontal Position Representation,
    The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
    <www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>`_
    """
    return E_ROTATION_MATRIX[axes]


def nthroot(x, n):
    """
    Returns the n'th root of x to machine precision

    Parameters
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


def deg(rad_angle):
    """
    Converts angle in radians to degrees.

    Parameters
    ----------
    rad_angle:
        angle in radians

    Returns
    -------
    deg_angle:
        angle in degrees

    See also
    --------
    rad
    """
    return rad2deg(rad_angle)


def rad(deg_angle):
    """
    Converts angle in degrees to radians.

    Parameters
    ----------
    deg_angle:
        angle in degrees

    Returns
    -------
    rad_angle:
        angle in radians

    See also
    --------
    deg
    """
    return deg2rad(deg_angle)


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
    current_norm = norm(vector, axis=0)
    idx = np.flatnonzero(current_norm == 0)
    unit_vector = vector / (current_norm + _TINY)

    unit_vector[:, idx] = 0 * norm_zero_vector
    unit_vector[0, idx] = 1 * norm_zero_vector
    return unit_vector


def mdot(a, b):
    """
    Returns multiple matrix multiplications of two arrays
      i.e.   dot(a, b)[i,j,k] = sum(a[i,:,j] * b[:,j,k])
    or
      np.concatenate([np.dot(a[...,i], b[...,i])[:, :, None]
                      for i in range(2)], axis=2)

    Examples
    --------
    3 x 3 x 2 times 3 x 3 x 2 array -> 3 x 2 x 2 array
    >>> import numpy as np
    >>> a = 1.0 * np.arange(18).reshape(3,3,2)
    >>> b = - a
    >>> t = np.concatenate([np.dot(a[...,i], b[...,i])[:, :, None]
    ...                    for i in range(2)], axis=2)
    >>> tt = mdot(a, b)
    >>> tt.shape
    (3, 3, 2)
    >>> np.allclose(t, tt)
    True

    3 x 3 x 2 times 3 x 1 array -> 3 x 1 x 2 array
    >>> t1 = np.concatenate([np.dot(a[...,i], b[:,0,0][:,None])[:,:,None]
    ...                    for i in range(2)], axis=2)

    >>> tt = mdot(a, b[:,0,0].reshape(-1,1))
    >>> tt.shape
    (3, 1, 2)
    >>> np.allclose(t1, tt)
    True

    3 x 3  times 3 x 3 array -> 3 x 3 array
    >>> tt0 = mdot(a[...,0], b[...,0])
    >>> tt0.shape
    (3, 3)
    >>> np.allclose(t[...,0], tt0)
    True

    3 x 3  times 3 x 1 array -> 3 x 1 array
    >>> tt0 = mdot(a[...,0], b[:,0,0][:,None])
    >>> tt0.shape
    (3, 1)
    >>> np.allclose(t[:,0,0][:,None], tt0)
    True

    3 x 3  times 3 x 2 array -> 3 x 1 x 2 array
    >>> tt0 = mdot(a[..., 0], b[:, :2, 0][:, None])
    >>> tt0.shape
    (3, 1, 2)
    >>> np.allclose(t[:,:2,0][:,None], tt0)
    True

    """
    return np.einsum('ij...,jk...->ik...', a, b)


def lat_lon2n_E(latitude, longitude, R_Ee=None):
    """
    Converts latitude and longitude to n-vector.

    Parameters
    ----------
    latitude, longitude: real scalars or vectors of length n.
        Geodetic latitude and longitude given in [rad]
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    n_E: 3 x n array
        n-vector(s) [no unit] decomposed in E.

    See also
    --------
    n_E2lat_lon
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    # Equation (3) from Gade (2010):
    nvec = np.vstack((sin(latitude),
                      sin(longitude) * cos(latitude),
                      -cos(longitude) * cos(latitude)))
    n_E = dot(R_Ee.T, nvec)
    return n_E


def n_E2lat_lon(n_E, R_Ee=None):
    """
    Converts n-vector to latitude and longitude.

    Parameters
    ----------
    n_E: 3 x n array
        n-vector [no unit] decomposed in E.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    latitude, longitude: real scalars or vectors of length n.
        Geodetic latitude and longitude given in [rad]

    See also
    --------
    lat_lon2n_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    _check_length_deviation(n_E)
    n_E0 = dot(R_Ee, n_E)

    # Equation (5) in Gade (2010):
    longitude = arctan2(n_E0[1, :], -n_E0[2, :])

    # Equation (6) in Gade (2010) (Robust numerical solution)
    equatorial_component = sqrt(n_E0[1, :]**2 + n_E0[2, :]**2)
    # vector component in the equatorial plane
    latitude = arctan2(n_E0[0, :], equatorial_component)
    # atan() could also be used since latitude is within [-pi/2,pi/2]

    # latitude=asin(n_E0[0] is a theoretical solution, but close to the Poles
    # it is ill-conditioned which may lead to numerical inaccuracies (and it
    # will give imaginary results for norm(n_E)>1)
    return latitude, longitude


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
    length_deviation = np.abs(norm(n_E[:, 0]) - 1)
    if length_deviation > limit:
        warnings.warn('n-vector should have unit length: '
                      'norm(n_E)~=1 ! Error is: {}'.format(length_deviation))


def n_E2R_EN(n_E, R_Ee=None):
    """
    Returns the rotation matrix R_EN from n-vector.

    Parameters
    ----------
    n_E: 3 x n array
        n-vector [no unit] decomposed in E
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    R_EN:  3 x 3 x n array
        The resulting rotation matrix [no unit] (direction cosine matrix).

    See also
    --------
    R_EN2n_E, n_E_and_wa2R_EL, R_EL2n_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    _check_length_deviation(n_E)
    n_E = unit(dot(R_Ee, n_E))

    # N coordinate frame (North-East-Down) is defined in Table 2 in Gade (2010)
    # Find z-axis of N (Nz):
    Nz_E = -n_E  # z-axis of N (down) points opposite to n-vector

    # Find y-axis of N (East)(remember that N is singular at Poles)
    # Equation (9) in Gade (2010):
    # Ny points perpendicular to the plane
    Ny_E_direction = cross([[1], [0], [0]], n_E, axis=0)
    # formed by n-vector and Earth's spin axis
    on_poles = np.flatnonzero(norm(Ny_E_direction, axis=0) == 0)
    Ny_E = unit(Ny_E_direction)
    Ny_E[:, on_poles] = array([[0], [1], [0]])  # selected y-axis direction

    # Find x-axis of N (North):
    Nx_E = cross(Ny_E, Nz_E, axis=0)  # Final axis found by right hand rule

    # Form R_EN from the unit vectors:
    # R_EN = dot(R_Ee.T, np.hstack((Nx_E, Ny_E, Nz_E)))
    Nxyz_E = np.hstack((Nx_E[:, None, ...],
                        Ny_E[:, None, ...],
                        Nz_E[:, None, ...]))
    R_EN = mdot(np.rollaxis(R_Ee, 1, 0), Nxyz_E)

    return np.squeeze(R_EN)


def n_E_and_wa2R_EL(n_E, wander_azimuth, R_Ee=None):
    """
    Returns rotation matrix R_EL from n-vector and wander azimuth angle.

    Parameters
    ----------
    n_E: 3 x n array
        n-vector [no unit] decomposed in E
    wander_azimuth: real scalar or array of length n
        Angle [rad] between L's x-axis and north, positive about L's z-axis.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    R_EL: 3 x 3 x n array
        The resulting rotation matrix.       [no unit]

    Notes
    -----
    When wander_azimuth=0, we have that N=L.
    (See Table 2 in Gade (2010) for details)

    See also
    --------
    R_EL2n_E, R_EN2n_E, n_E2R_EN
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    latitude, longitude = n_E2lat_lon(n_E, R_Ee)

    # Reference: See start of Section 5.2 in Gade (2010):
    R_EL = mdot(R_Ee.T, xyz2R(longitude, -latitude, wander_azimuth))
    return np.squeeze(R_EL)


class _Nvector2ECEFvector(object):
    __doc__ = """
    Converts n-vector to Cartesian position vector in meters.

    Parameters
    ----------
    n_EB_E:  3 x n array
        n-vector(s) [no unit] of position B, decomposed in E.
    depth:  1 x n array
        Depth(s) [m] of system B, relative to the ellipsoid (depth = -height)
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    p_EB_E:  3 x n array
        Cartesian position vector(s) from E to B, decomposed in E.

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, n_EB_E. The function converts
    to cartesian position vector ("ECEF-vector"), p_EB_E, in meters.
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    Examples
    --------

    {0}

    See also
    --------
    p_EB_E2n_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E
    """.format(_examples.get_examples([4], OO=False))


@use_docstring_from(_Nvector2ECEFvector)
def n_EB_E2p_EB_E(n_EB_E, depth=0, a=6378137, f=1.0 / 298.257223563, R_Ee=None):
    if R_Ee is None:
        R_Ee = E_rotation()
    _check_length_deviation(n_EB_E)

    n_EB_E = unit(dot(R_Ee, n_EB_E))
    b = a * (1 - f)  # semi-minor axis

    # The following code implements equation (22) in Gade (2010):
    scale = np.vstack((1,
                       (1 - f),
                       (1 - f)))
    denominator = norm(n_EB_E / scale, axis=0)

    # We first calculate the position at the origin of coordinate system L,
    # which has the same n-vector as B (n_EL_E = n_EB_E),
    # but lies at the surface of the Earth (z_EL = 0).

    p_EL_E = b / denominator * n_EB_E / scale**2
    p_EB_E = dot(R_Ee.T, p_EL_E - n_EB_E * depth)

    return p_EB_E


class _ECEFvector2Nvector(object):
    __doc__ = """
    Converts Cartesian position vector in meters to n-vector.

    Parameters
    ----------
    p_EB_E:  3 x n array
        Cartesian position vector(s) from E to B, decomposed in E.
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    n_EB_E:  3 x n array
        n-vector(s) [no unit] of position B, decomposed in E.
    depth:  1 x n array
        Depth(s) [m] of system B, relative to the ellipsoid (depth = -height)


    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as cartesian position vector p_EB_E, in meters.
    ("ECEF-vector"). The function converts to n-vector, n_EB_E and its
    depth, depth.
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    Examples
    --------

    {0}

    See also
    --------
    n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E

    """.format(_examples.get_examples([3], OO=False))


@use_docstring_from(_ECEFvector2Nvector)
def p_EB_E2n_EB_E(p_EB_E, a=6378137, f=1.0 / 298.257223563, R_Ee=None):
    if R_Ee is None:
        # R_Ee selects correct E-axes, see E_rotation for details
        R_Ee = E_rotation()
    p_EB_E = dot(R_Ee, p_EB_E)

    # e_2 = eccentricity**2
    e_2 = 2 * f - f**2  # = 1-b**2/a**2

    # The following code implements equation (23) from Gade (2010):
    R_2 = p_EB_E[1, :]**2 + p_EB_E[2, :]**2
    R = sqrt(R_2)   # R = component of p_EB_E in the equatorial plane

    p = R_2 / a**2
    q = (1 - e_2) / (a**2) * p_EB_E[0, :]**2
    r = (p + q - e_2**2) / 6

    s = e_2**2 * p * q / (4 * r**3)
    t = nthroot((1 + s + sqrt(s * (2 + s))), 3)
    # t = (1 + s + sqrt(s * (2 + s)))**(1. / 3)
    u = r * (1 + t + 1. / t)
    v = sqrt(u**2 + e_2**2 * q)

    w = e_2 * (u + v - q) / (2 * v)
    k = sqrt(u + v + w**2) - w
    d = k * R / (k + e_2)

    temp0 = sqrt(d**2 + p_EB_E[0, :]**2)
    # Calculate height:
    height = (k + e_2 - 1) / k * temp0

    temp1 = 1. / temp0
    temp2 = temp1 * k / (k + e_2)

    n_EB_E_x = temp1 * p_EB_E[0, :]
    n_EB_E_y = temp2 * p_EB_E[1, :]
    n_EB_E_z = temp2 * p_EB_E[2, :]

    n_EB_E = np.vstack((n_EB_E_x, n_EB_E_y, n_EB_E_z))
    n_EB_E = unit(dot(R_Ee.T, n_EB_E))  # Ensure unit length

    depth = -height
    return n_EB_E, depth


class _DeltaFromPositionAtoB(object):
    __doc__ = """
    Returns the delta vector from position A to B decomposed in E.

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.
    z_EA, z_EB:  1 x n array
        Depth(s) [m] of system A and B, relative to the ellipsoid.
        (z_EA = -height, z_EB = -height)
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    p_AB_E:  3 x n array
        Cartesian position vector(s) from A to B, decomposed in E.

    Notes
    -----
    The n-vectors for positions A (n_EA_E) and B (n_EB_E) are given. The
    output is the delta vector from A to B (p_AB_E).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    Examples
    --------

    {0}


    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E

    """.format(_examples.get_examples([1], False))


@use_docstring_from(_DeltaFromPositionAtoB)
def n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=6378137,
                             f=1.0 / 298.257223563, R_Ee=None):

    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = n_EB_E2p_EB_E(n_EB_E, z_EB, a, f, R_Ee)
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


def n_EA_E_and_p_AB_E2n_EB_E(n_EA_E, p_AB_E, z_EA=0, a=6378137,
                             f=1.0 / 298.257223563, R_Ee=None):
    """
    Returns position B from position A and delta.

    Parameters
    ----------
    n_EA_E:  3 x n array
        n-vector(s) [no unit] of position A, decomposed in E.
    p_AB_E:  3 x n array
        Cartesian position vector(s) from A to B, decomposed in E.
    z_EA:  1 x n array
        Depth(s) [m] of system A, relative to the ellipsoid. (z_EA = -height)
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    n_EB_E:  3 x n array
        n-vector(s) [no unit] of position B, decomposed in E.
    z_EB:  1 x n array
        Depth(s) [m] of system B, relative to the ellipsoid.
        (z_EB = -height)

    Notes
    -----
    The n-vector for position A (n_EA_E) and the position-vector from position
    A to position B (p_AB_E) are given. The output is the n-vector of position
    B (n_EB_E) and depth of B (z_EB).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    See also
    --------
    n_EA_E_and_n_EB_E2p_AB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()

    # Function 2. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = p_EA_E + p_AB_E
    n_EB_E, z_EB = p_EB_E2n_EB_E(p_EB_E, a, f, R_Ee)
    return n_EB_E, z_EB


def R2xyz(R_AB):
    """
    Returns the angles about new axes in the xyz-order from a rotation matrix.

    Parameters
    ----------
    R_AB: 3 x 3 x n array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = mdot(R_AB, v_B)

    Returns
    -------
    x, y, z: real scalars or array of length n.
        Angles [rad] of rotation about new axes.

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

    See also
    --------
    xyz2R, R2zyx, xyz2R
    """
    z = arctan2(-R_AB[0, 1, ...], R_AB[0, 0, ...])  # atan2: [-pi pi]
    x = arctan2(-R_AB[1, 2, ...], R_AB[2, 2, ...])

    sin_y = R_AB[0, 2, ...]

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = sqrt((R_AB[0, 0, ...]**2 + R_AB[0, 1, ...]**2
                  + R_AB[1, 2, ...]**2 + R_AB[2, 2, ...]**2) / 2)

    y = arctan2(sin_y, cos_y)
    return x, y, z


def R2zyx(R_AB):
    """
    Returns the angles about new axes in the zxy-order from a rotation matrix.

    Parameters
    ----------
    R_AB:  3x3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

    Returns
    -------
    z, y, x: real scalars
        Angles [rad] of rotation about new axes.

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

    See also
    --------
    zyx2R, xyz2R, R2xyz
    """
    x, y, z = R2xyz(np.rollaxis(R_AB, 1, 0))
    return -z, -y, -x


def R_EL2n_E(R_EL):
    """
    Returns n-vector from the rotation matrix R_EL.

    Parameters
    ----------
    R_EL: 3 x 3 x n array
        Rotation matrix (direction cosine matrix) [no unit]

    Returns
    -------
    n_E: 3 x n array
        n-vector(s) [no unit] decomposed in E.

    See also
    --------
    R_EN2n_E, n_E_and_wa2R_EL, n_E2R_EN
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = mdot(R_EL, np.vstack((0, 0, -1)))
    return n_E.reshape(3, -1)


def R_EN2n_E(R_EN):
    """
    Returns n-vector from the rotation matrix R_EN.

    Parameters
    ----------
    R_EN: 3 x 3 x n array
        Rotation matrix (direction cosine matrix) [no unit]

    Returns
    -------
    n_E: 3 x n array
        n-vector [no unit] decomposed in E.

    See also
    --------
    n_E2R_EN, R_EL2n_E, n_E_and_wa2R_EL
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    return R_EL2n_E(R_EN)


def _atleast_3d(x, y, z):
    """
    Example
    -------
    >>> from nvector._core import _atleast_3d
    >>> for arr in _atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
     ...     print(arr, arr.shape)
    [[[1 2]]] (1, 1, 2)
    [[[[1 2]]]] (1, 1, 1, 2)
    [[[[[1 2]]]]] (1, 1, 1, 1, 2)

     >>> for arr in _atleast_3d([[1], [2]], [[[1], [2]]], [[[[1], [2]]]]):
     ...     print(arr, arr.shape)
    [[[[1]
       [2]]]] (1, 1, 2, 1)
    [[[[[1]
        [2]]]]] (1, 1, 1, 2, 1)
    [[[[[[1]
         [2]]]]]] (1, 1, 1, 1, 2, 1)
    """
    x, y, z = np.atleast_1d(x, y, z)
    return x[None, None, :], y[None, None, :], z[None, None, :]


def xyz2R(x, y, z):
    """
    Returns rotation matrix from 3 angles about new axes in the xyz-order.

    Parameters
    ----------
    x,y,z: real scalars or array of lengths n
        Angles [rad] of rotation about new axes.

    Returns
    -------
    R_AB: 3 x 3 x n array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = mdot(R_AB, v_B)

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

    See also
    --------
    R2xyz, zyx2R, R2zyx
    """
    x, y, z = _atleast_3d(x, y, z)
    sx, sy, sz = sin(x), sin(y), sin(z)
    cx, cy, cz = cos(x), cos(y), cos(z)

    R_AB = array(([cy * cz, -cy * sz, sy],
                  [sy * sx * cz + cx * sz, -sy * sx * sz + cx * cz, -cy * sx],
                  [-sy * cx * cz + sx * sz, sy * cx * sz + sx * cz, cy * cx]))

    return np.squeeze(R_AB)


def zyx2R(z, y, x):
    """
    Returns rotation matrix from 3 angles about new axes in the zyx-order.

    Parameters
    ----------
    z, y, x: real scalars or arrays of lenths n
        Angles [rad] of rotation about new axes.

    Returns
    -------
    R_AB: 3 x 3 x n array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = mdot(R_AB, v_B)

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

    See also
    --------
    R2zyx, xyz2R, R2xyz
    """
    x, y, z = _atleast_3d(x, y, z)
    sx, sy, sz = sin(x), sin(y), sin(z)
    cx, cy, cz = cos(x), cos(y), cos(z)

    R_AB = array([[cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy * cx],
                  [sz * cy, cz * cx + sz * sy * sx, - cz * sx + sz * sy * cx],
                  [-sy, cy * sx, cy * cx]])

    return np.squeeze(R_AB)


def interpolate(path, ti):
    """
    Returns the interpolated point along the path

    Parameters
    ----------
    path: tuple of n-vectors (positionA, positionB)

    ti: real scalar
        interpolation time assuming position A and B is at t0=0 and t1=1,
        respectively.

    Returns
    -------
    point: Nvector
        point of interpolation along path
    """

    n_EB_E_t0, n_EB_E_t1 = path
    n_EB_E_ti = unit(n_EB_E_t0 + ti * (n_EB_E_t1 - n_EB_E_t0),
                     norm_zero_vector=np.nan)
    return n_EB_E_ti


class _Intersect(object):
    __doc__ = """Returns the intersection(s) between the great circles of the two paths

    Parameters
    ----------
    path_a, path_b: tuple of 2 n-vectors
        defining path A and path B, respectively.
        Path A and B has shape 2 x 3 x n and 2 x 3 x m, respectively.

    Returns
    -------
    n_EC_E : array of shape 3 x max(n, m)
        n-vector(s) [no unit] of position C decomposed in E.
        point(s) of intersection between paths.

    Examples
    --------

    {0}

    """.format(_examples.get_examples([9], OO=False))


@use_docstring_from(_Intersect)
def intersect(path_a, path_b):
    n_EA1_E, n_EA2_E = path_a
    n_EB1_E, n_EB2_E = path_b
    # Find the intersection between the two paths, n_EC_E:
    n_EC_E_tmp = unit(cross(cross(n_EA1_E, n_EA2_E, axis=0),
                            cross(n_EB1_E, n_EB2_E, axis=0), axis=0),
                      norm_zero_vector=np.nan)

    # n_EC_E_tmp is one of two solutions, the other is -n_EC_E_tmp. Select
    # the one that is closet to n_EA1_E, by selecting sign from the dot
    # product between n_EC_E_tmp and n_EA1_E:
    n_EC_E = np.sign(dot(n_EC_E_tmp.T, n_EA1_E)) * n_EC_E_tmp
    if np.any(np.isnan(n_EC_E)):
        warnings.warn('Paths are Equal. Intersection point undefined. '
                      'NaN returned.')
    return n_EC_E


def great_circle_normal(n_EA_E, n_EB_E):
    """
    Returns the unit normal(s) to the great circle(s)

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.

    """
    return unit(cross(n_EA_E, n_EB_E, axis=0), norm_zero_vector=np.nan)


def _euclidean_cross_track_distance(sin_theta, radius=1):
    return sin_theta * radius


def _great_circle_cross_track_distance(sin_theta, radius=1):
    return np.arcsin(sin_theta) * radius
    # ill conditioned for small angles:
    # return (np.arccos(-sin_theta) - np.pi / 2) * radius


class _CrossTrackDistance(object):
    __doc__ = """Returns  cross track distance between path A and position B.

    Parameters
    ----------
    path: tuple of 2 n-vectors
        2 n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to measure the cross track distance to.
    method: string
        defining distance calculated. Options are: 'greatcircle' or 'euclidean'
    radius: real scalar
        radius of sphere. (default 6371009.0)

    Returns
    -------
    distance : array of length max(n, m)
        cross track distance(s)

    Examples
    --------

    {0}

    """.format(_examples.get_examples([10], OO=False))


@use_docstring_from(_CrossTrackDistance)
def cross_track_distance(path, n_EB_E, method='greatcircle',
                         radius=6371009.0):

    c_E = great_circle_normal(path[0], path[1])
    sin_theta = -np.dot(c_E.T, n_EB_E).ravel()
    if method[0].lower() == 'e':
        return _euclidean_cross_track_distance(sin_theta, radius)
    return _great_circle_cross_track_distance(sin_theta, radius)


class _OnGreatCircle(object):
    __doc__ = """ True if position B is on great circle through path A.

    Parameters
    ----------
    path: tuple of 2 n-vectors
        2 n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to check to.
    radius: real scalar
        radius of sphere. (default 6371009.0)
    rtol, atol: real scalars
        defining relative and absolute tolerance

    Returns
    -------
    on : bool array of length max(n, m)
        True if position B is on great circle through path A.

    Examples
    --------

    {0}

    """.format(_examples.get_examples([10], OO=False))


@use_docstring_from(_OnGreatCircle)
def on_great_circle(path, n_EB_E, radius=6371009.0, rtol=1e-6, atol=1e-8):
    distance = np.abs(cross_track_distance(path, n_EB_E, radius=radius))
    return np.isclose(distance, 0, rtol, atol)


class _OnGreatCirclePath(object):
    __doc__ = """ True if position B is on great circle and between endpoints of path A.

    Parameters
    ----------
    path: tuple of 2 n-vectors
        2 n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to measure the cross track distance to.
    radius: real scalar
        radius of sphere. (default 6371009.0)
    rtol, atol: real scalars
        defining relative and absolute tolerance

    Returns
    -------
    on : bool array of length max(n, m)
        True if position B is on great circle and between endpoints of path A.

    Examples
    --------

    {0}

    """.format(_examples.get_examples([10], OO=False))


@use_docstring_from(_OnGreatCirclePath)
def on_great_circle_path(path, n_EB_E, radius=6371009.0, rtol=1e-6, atol=1e-8):
    n_EA1_E, n_EA2_E = path
    scale = norm(n_EA2_E - n_EA1_E, axis=0)
    ti1 = norm(n_EB_E - n_EA1_E, axis=0) / scale
    ti2 = norm(n_EB_E - n_EA2_E, axis=0) / scale
    return (ti1 <= 1) & (ti2 <= 1) & on_great_circle(path, n_EB_E, radius,
                                                     rtol, atol)


class _ClosestPointOnGreatCircle(object):
    __doc__ = """Returns closest point C on great circle path A to position B.

    Parameters
    ----------
    path: tuple of 2 n-vectors of 3 x n arrays
        2 n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to find the closest point to.

    Returns
    -------
    n_EC_E:  3 x max(m, n) array
        n-vector(s) of closest position C on great circle path A

    Examples
    --------

    {0}

    """.format(_examples.get_examples([10], OO=False))


@use_docstring_from(_ClosestPointOnGreatCircle)
def closest_point_on_great_circle(path, n_EB_E):

    n_EA1_E, n_EA2_E = path

    c1 = cross(n_EA1_E, n_EA2_E, axis=0)
    c2 = cross(n_EB_E, c1, axis=0)
    n_EC_E = unit(cross(c1, c2, axis=0))
    return n_EC_E


class _GreatCircleDistance(object):
    __doc__ = """Returns great circle distance between positions A and B

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.
    radius: real scalar
        radius of sphere.

    Formulae is given by equation (16) in Gade (2010) and is well
    conditioned for all angles.

    Examples
    --------

    {0}

    """.format(_examples.get_examples([5], OO=False))


@use_docstring_from(_GreatCircleDistance)
def great_circle_distance(n_EA_E, n_EB_E, radius=6371009.0):

    s_AB = np.arctan2(norm(np.cross(n_EA_E, n_EB_E, axis=0), axis=0),
                      dot(n_EA_E.T, n_EB_E)) * radius

    # ill conditioned for small angles:
    # s_AB_version1 = arccos(dot(n_EA_E,n_EB_E))*radius

    # ill-conditioned for angles near pi/2 (and not valid above pi/2)
    # s_AB_version2 = arcsin(norm(cross(n_EA_E,n_EB_E)))*radius
    return s_AB.ravel()


class _EuclideanDistance(object):
    __doc__ = """Returns Euclidean distance between positions A and B

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.
    radius: real scalar
        radius of sphere.

    Examples
    --------

    {0}
    """.format(_examples.get_examples([5], OO=False))


@use_docstring_from(_EuclideanDistance)
def euclidean_distance(n_EA_E, n_EB_E, radius=6371009.0):
    d_AB = norm(n_EB_E - n_EA_E, axis=0) * radius
    return d_AB.ravel()


def n_EA_E_and_n_EB_E2azimuth(n_EA_E, n_EB_E, a=6378137, f=1.0 / 298.257223563,
                              R_Ee=None):
    """
    Returns azimuth from A to B, relative to North:

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, respectively,
        decomposed in E.
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    azimuth: n array
        Angle [rad] the line makes with a meridian, taken clockwise from north.
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    # Step2: Find p_AB_E (delta decomposed in E).
    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=a, f=f,
                                      R_Ee=R_Ee)

    # Step3: Find R_EN for position A:
    R_EN = n_E2R_EN(n_EA_E, R_Ee=R_Ee)

    # Step4: Find p_AB_N
    # p_AB_N = dot(R_EN.T, p_AB_E)
    p_AB_N = mdot(np.rollaxis(R_EN, 1, 0), p_AB_E[:, None, ...]).reshape(3, -1)
    # (Note the transpose of R_EN: The "closest-rule" says that when
    # decomposing, the frame in the subscript of the rotation matrix that
    # is closest to the vector, should equal the frame where the vector is
    # decomposed. Thus the calculation np.dot(R_NE, p_AB_E) is correct,
    # since the vector is decomposed in E, and E is closest to the vector.
    # In the example we only had R_EN, and thus we must transpose it:
    # R_EN'=R_NE)

    # Step5: Also find the direction (azimuth) to B, relative to north:
    return arctan2(p_AB_N[1], p_AB_N[0])


class _PositionBFromAzimuthAndDistanceFromPositionA(object):
    __doc__ = """
    Returns position B from azimuth and distance from position A

    Parameters
    ----------
    n_EA_E:  3 x n array
        n-vector(s) [no unit] of position A decomposed in E.
    distance_rad: n, array
        great circle distance [rad] from position A to B
    azimuth: n array
        Angle [rad] the line makes with a meridian, taken clockwise from north.

    Returns
    -------
    n_EB_E:  3 x n array
        n-vector(s) [no unit] of position B decomposed in E.

    Examples
    --------

    {0}

    """.format(_examples.get_examples([8], OO=False))


@use_docstring_from(_PositionBFromAzimuthAndDistanceFromPositionA)
def n_EA_E_distance_and_azimuth2n_EB_E(n_EA_E, distance_rad, azimuth,
                                       R_Ee=None):

    if R_Ee is None:
        R_Ee = E_rotation()
    # Step1: Find unit vectors for north and east:
    k_east_E = unit(cross(dot(R_Ee.T, [[1], [0], [0]]), n_EA_E, axis=0))
    k_north_E = cross(n_EA_E, k_east_E, axis=0)

    # Step2: Find the initial direction vector d_E:
    d_E = k_north_E * cos(azimuth) + k_east_E * sin(azimuth)

    # Step3: Find n_EB_E:
    n_EB_E = n_EA_E * cos(distance_rad) + d_E * sin(distance_rad)
    return n_EB_E


class _MeanHorizontalPosition(object):
    __doc__ = """
    Returns the n-vector of the horizontal mean position.

    Parameters
    ----------
    n_EB_E:  3 x n array
        n-vectors [no unit] of positions Bi, decomposed in E.

    Returns
    -------
    p_EM_E:  3 x 1 array
        n-vector [no unit] of the mean positions of all Bi, decomposed in E.

    Examples
    --------

    {0}

    """.format(_examples.get_examples([7], OO=False))


@use_docstring_from(_MeanHorizontalPosition)
def mean_horizontal_position(n_EB_E):
    n_EM_E = unit(np.sum(n_EB_E, axis=1).reshape((3, 1)))
    return n_EM_E


if __name__ == "__main__":
    test_docstrings(__file__)
