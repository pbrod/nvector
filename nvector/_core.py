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
import numpy as np
from numpy import rad2deg, deg2rad, arctan2, sin, cos, array, cross, dot, sqrt
from numpy.linalg import norm
import warnings
from nvector import _examples
from nvector._examples import use_docstring_from


__all__ = ['select_ellipsoid', 'E_rotation',
           'unit', 'deg', 'rad', 'nthroot',
           'lat_lon2n_E', 'n_E2lat_lon',
           'n_EA_E_and_n_EB_E2p_AB_E', 'n_EA_E_and_p_AB_E2n_EB_E',
           'p_EB_E2n_EB_E', 'n_EB_E2p_EB_E',
           'n_EA_E_distance_and_azimuth2n_EB_E',
           'n_EA_E_and_n_EB_E2azimuth',
           'great_circle_distance', 'euclidean_distance',
           'mean_horizontal_position',
           'R2xyz', 'xyz2R', 'R2zyx', 'zyx2R',
           'n_E_and_wa2R_EL', 'n_E2R_EN', 'R_EL2n_E', 'R_EN2n_E'
           ]

E_ROTATION_MATRIX = dict(e=array([[0, 0, 1],
                                  [0, 1, 0],
                                  [-1, 0, 0]]),
                         E=np.eye(3))

_EPS = np.finfo(float).eps  # machine precision (machine epsilon)


ELLIPSOID = {1: ({'a': 6377563.3960, 'f': 1.0/299.3249646}, 'Airy 1858'),
             2: ({'a': 6377340.189, 'f': 1.0/299.3249646}, 'Airy Modified'),
             3: ({'a': 6378160, 'f': 1.0/298.25}, 'Australian National'),
             4: ({'a': 6377397.155, 'f': 1.0/299.1528128}, 'Bessel 1841'),
             5: ({'a': 6378249.145, 'f': 1.0/293.465}, 'Clarke 1880'),
             6: ({'a': 6377276.345, 'f': 1.0/300.8017}, 'Everest 1830'),
             7: ({'a': 6377304.063, 'f': 1.0/300.8017}, 'Everest Modified'),
             8: ({'a': 6378166.0, 'f': 1.0/298.3}, 'Fisher 1960'),
             9: ({'a': 6378150.0, 'f': 1.0/298.3}, 'Fisher 1968'),
             10: ({'a': 6378270.0, 'f': 1.0/297}, 'Hough 1956'),
             11: ({'a': 6378388.0, 'f': 1.0/297}, 'International (Hayford)'),
             12: ({'a': 6378245.0, 'f': 1.0/298.3}, 'Krassovsky 1938'),
             13: ({'a': 6378145., 'f': 1.0/298.25}, 'NWL-9D  (WGS 66)'),
             14: ({'a': 6378160., 'f': 1.0/298.25}, 'South American 1969'),
             15: ({'a': 6378136, 'f': 1.0/298.257},
                  'Soviet Geod. System 1985'),
             16: ({'a': 6378135., 'f': 1.0/298.26}, 'WGS 72'),
             17: ({'a': 6378206.4, 'f': 1.0/294.9786982138},
                  'Clarke 1866    (NAD27)'),
             18: ({'a': 6378137.0, 'f': 1.0/298.257223563},
                  'GRS80 / WGS84  (NAD83)')}

ELLIPSOID_IX = {'airy1858': 1, 'airymodified': 2, 'australiannational': 3,
                'everest1830': 6, 'everestmodified': 7, 'krassovsky': 12,
                'krassovsky1938': 12, 'fisher1968': 9, 'fisher1960': 8,
                'international': 11, 'hayford': 11,
                'clarke1866': 17, 'nad27': 17, 'bessel': 4,
                'bessel1841': 4, 'grs80': 18, 'wgs84': 18, 'nad83': 18,
                'sovietgeod.system1985': 15, 'wgs72': 16,
                'hough1956': 10, 'hough': 10, 'nwl-9d': 13, 'wgs66': 13,
                'southamerican1969': 14,  'clarke1880': 5}


def select_ellipsoid(name):
    """
    Return semi-major axis (a), flattening (f) and name of ellipsoid

    Parameters
    ----------
    name : string
        name of ellipsoid. Valid options are:
        'airy1858', 'airymodified', 'australiannational', 'everest1830',
        'everestmodified', 'krassovsky', 'krassovsky1938', 'fisher1968',
        'fisher1960', 'international', 'hayford', 'clarke1866', 'nad27',
        'bessel', 'bessel1841', 'grs80', 'wgs84', 'nad83',
        'sovietgeod.system1985', 'wgs72', 'hough1956', 'hough', 'nwl-9d',
        'wgs66', 'southamerican1969',  'clarke1880'.

    Examples
    --------
    >>> import nvector as nv
    >>> nv.select_ellipsoid(name='wgs84')
    (6378137.0, 0.0033528106647474805, 'GRS80 / WGS84  (NAD83)')
    """
    msg = """
    Other Ellipsoids.'
    -----------------'
    '
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
    11) International (Hayford)
    12) Krassovsky 1938
    13) NWL-9D (WGS 66)
    14) South American 1969
    15) Soviet Geod. System 1985
    16) WGS 72
    17) Clarke 1866    (NAD27)
    18) GRS80 / WGS84  (NAD83)
    '
    Enter choice :
    """

    if name:
        option = ELLIPSOID_IX.get(name.lower().replace(' ', ''), name)
    else:
        option = input(msg)
    ellipsoid, fullname = ELLIPSOID[option]
    return ellipsoid['a'], ellipsoid['f'], fullname


def E_rotation(axes='e'):
    """
    Return rotation matrix R_Ee defining the axes of the coordinate frame E.

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
    R_Ee : 2d array
        rotation matrix defining the axes of the coordinate frame E as
        described in Table 2 in Gade (2010)

    R_Ee controls the axes of the coordinate frame E (Earth-Centred,
    Earth-Fixed, ECEF) used by the other functions in this library

    Examples
    --------
    >>> import nvector as nv
    >>> nv.E_rotation(axes='e')
    array([[ 0,  0,  1],
           [ 0,  1,  0],
           [-1,  0,  0]])
    >>> nv.E_rotation(axes='E')
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    Reference
    ---------
    Gade, K. (2010). `A Nonsingular Horizontal Position Representation,
    The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
    <www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>`_
    """
    return E_ROTATION_MATRIX[axes]


def nthroot(x, n):
    """
    Return the n'th root of x to machine precision

    Parameters
    x, n

    Examples
    --------
    >>> import nvector as nv
    >>> nv.nthroot(27.0, 3)
    array(3.0)

    """
    y = x**(1./n)
    return np.where((x != 0) & (_EPS * np.abs(x) < 1),
                    y - (y**n-x)/(n*y**(n-1)), y)


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

    Examples
    --------
    >>> import nvector as nv
    >>> nv.unit([[1],[1],[1]])
    array([[ 0.57735027],
           [ 0.57735027],
           [ 0.57735027]])

    """
    current_norm = norm(vector, axis=0)
    unit_vector = vector / current_norm
    idx = np.flatnonzero(current_norm == 0)

    unit_vector[:, idx] = 0 * norm_zero_vector
    unit_vector[0, idx] = 1 * norm_zero_vector
    return unit_vector


def lat_lon2n_E(latitude, longitude, R_Ee=None):
    """
    Converts latitude and longitude to n-vector.

    Parameters
    ----------
    latitude, longitude: real scalars or vectors of length n.
        Geodetic latitude and longitude given in [rad]
    R_Ee : 2d array
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
    n_E = np.dot(R_Ee.T, nvec)
    return n_E


def n_E2lat_lon(n_E, R_Ee=None):
    """
    Converts n-vector to latitude and longitude.

    Parameters
    ----------
    n_E: 3 x n array
        n-vector [no unit] decomposed in E.
    R_Ee : 2d array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    latitude, longitude: real scalars or vectors of lengt n.
        Geodetic latitude and longitude given in [rad]

    See also
    --------
    lat_lon2n_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    _check_length_deviation(n_E)
    n_E = np.dot(R_Ee, n_E)

    # Equation (5) in Gade (2010):
    longitude = arctan2(n_E[1, :], -n_E[2, :])

    # Equation (6) in Gade (2010) (Robust numerical solution)
    equatorial_component = sqrt(n_E[1, :]**2 + n_E[2, :]**2)
    # vector component in the equatorial plane
    latitude = arctan2(n_E[0, :], equatorial_component)
    # atan() could also be used since latitude is within [-pi/2,pi/2]

    # latitude=asin(n_E[0] is a theoretical solution, but close to the Poles
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
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E
    R_Ee : 2d array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    R_EN:  3 x 3 array
        The resulting rotation matrix [no unit] (direction cosine matrix).

    See also
    --------
    R_EN2n_E, n_E_and_wa2R_EL, R_EL2n_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    _check_length_deviation(n_E)
    n_E = unit(np.dot(R_Ee, n_E))

    # N coordinate frame (North-East-Down) is defined in Table 2 in Gade (2010)
    # Find z-axis of N (Nz):
    Nz_E = -n_E  # z-axis of N (down) points opposite to n-vector

    # Find y-axis of N (East)(remember that N is singular at Poles)
    # Equation (9) in Gade (2010):
    # Ny points perpendicular to the plane
    Ny_E_direction = cross([[1], [0], [0]], n_E, axis=0)
    # formed by n-vector and Earth's spin axis
    outside_poles = (norm(Ny_E_direction) != 0)
    if outside_poles:
        Ny_E = unit(Ny_E_direction)
    else:  # Pole position:
        Ny_E = array([[0], [1], [0]])  # selected y-axis direction

    # Find x-axis of N (North):
    Nx_E = cross(Ny_E, Nz_E, axis=0)  # Final axis found by right hand rule

    # Form R_EN from the unit vectors:
    R_EN = dot(R_Ee.T, np.hstack((Nx_E, Ny_E, Nz_E)))

    return R_EN


def n_E_and_wa2R_EL(n_E, wander_azimuth, R_Ee=None):
    """
    Returns rotation matrix R_EL from n-vector and wander azimuth angle.

    R_EL = n_E_and_wa2R_EL(n_E,wander_azimuth) Calculates the rotation matrix
    (direction cosine matrix) R_EL using n-vector (n_E) and the wander
    azimuth angle.
    When wander_azimuth=0, we have that N=L (See Table 2 in Gade (2010) for
    details)

    Parameters
    ----------
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E
    wander_azimuth: real scalar
        Angle [rad] between L's x-axis and north, positive about L's z-axis.
    R_Ee : 2d array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    R_EL: 3 x 3 array
        The resulting rotation matrix.       [no unit]

    See also
    --------
    R_EL2n_E, R_EN2n_E, n_E2R_EN
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    latitude, longitude = n_E2lat_lon(n_E, R_Ee)

    # Reference: See start of Section 5.2 in Gade (2010):
    R_EL = dot(R_Ee.T, xyz2R(longitude, -latitude, wander_azimuth))
    return R_EL


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
    R_Ee : 2d array
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
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    Examples
    --------

    {}

    See also
    --------
    p_EB_E2n_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E
    """.format(_examples.get_examples([4], OO=False))


@use_docstring_from(_Nvector2ECEFvector)
def n_EB_E2p_EB_E(n_EB_E, depth=0, a=6378137, f=1.0/298.257223563, R_Ee=None):
    if R_Ee is None:
        R_Ee = E_rotation()
    _check_length_deviation(n_EB_E)

    n_EB_E = unit(dot(R_Ee, n_EB_E))
    if depth is None:
        depth = np.zeros((1, np.shape(n_EB_E)[1]))

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
    R_Ee : 2d array
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

    {}

    See also
    --------
    n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E

    """.format(_examples.get_examples([3], OO=False))


@use_docstring_from(_ECEFvector2Nvector)
def p_EB_E2n_EB_E(p_EB_E, a=6378137, f=1.0/298.257223563, R_Ee=None):
    if R_Ee is None:
        R_Ee = E_rotation()
    p_EB_E = dot(R_Ee, p_EB_E)
    # R_Ee selects correct E-axes, see R_Ee.m for details

    # e_2 = eccentricity**2
    e_2 = 2 * f - f**2  # = 1-b**2/a**2

    # The following code implements equation (23) from Gade (2010):
    R_2 = p_EB_E[1, :]**2 + p_EB_E[2, :]**2
    R = sqrt(R_2)   # R = component of p_EB_E in the equatorial plane

    p = R_2 / a**2
    q = (1 - e_2) / (a**2) * p_EB_E[0, :]**2
    r = (p + q - e_2**2) / 6

    s = e_2**2 * p * q / (4 * r**3)
    t = nthroot((1 + s + sqrt(s*(2+s))), 3)
    # t = (1 + s + sqrt(s * (2 + s)))**(1. / 3)
    u = r * (1 + t + 1. / t)
    v = sqrt(u**2 + e_2**2 * q)

    w = e_2 * (u + v - q) / (2 * v)
    k = sqrt(u + v + w**2) - w
    d = k * R / (k + e_2)

    # Calculate height:
    height = (k + e_2 - 1) / k * sqrt(d**2 + p_EB_E[0, :]**2)

    temp = 1. / sqrt(d**2 + p_EB_E[0, :]**2)

    n_EB_E_x = temp * p_EB_E[0, :]
    n_EB_E_y = temp * k / (k + e_2) * p_EB_E[1, :]
    n_EB_E_z = temp * k / (k + e_2) * p_EB_E[2, :]

    n_EB_E = np.vstack((n_EB_E_x, n_EB_E_y, n_EB_E_z))
    n_EB_E = unit(dot(R_Ee.T, n_EB_E))  # Ensure unit length:

    depth = -height
    return n_EB_E, depth


class _DeltaFromPositionAtoB(object):
    __doc__ = """
    Return the delta vector from position A to B.

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
    R_Ee : 2d array
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

    {}


    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E

    """.format(_examples.get_examples([1], False))


@use_docstring_from(_DeltaFromPositionAtoB)
def n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=6378137,
                             f=1.0/298.257223563, R_Ee=None):

    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = n_EB_E2p_EB_E(n_EB_E, z_EB, a, f, R_Ee)
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


def n_EA_E_and_p_AB_E2n_EB_E(n_EA_E, p_AB_E, z_EA=0, a=6378137,
                             f=1.0/298.257223563, R_Ee=None):
    """
    Return position B from position A and delta.

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
    R_Ee : 2d array
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
    R_AB: 3x3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

    Returns
    -------
    x, y, z: real scalars
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
    z = arctan2(-R_AB[0, 1], R_AB[0, 0])  # atan2: [-pi pi]
    x = arctan2(-R_AB[1, 2], R_AB[2, 2])

    sin_y = R_AB[0, 2]

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = sqrt((R_AB[0, 0]**2 + R_AB[0, 1]**2 +
                  R_AB[1, 2]**2 + R_AB[2, 2]**2)/2)

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

    z = arctan2(R_AB[1, 0], R_AB[0, 0])  # atan2: [-pi pi]
    x = arctan2(R_AB[2, 1], R_AB[2, 2])

    sin_y = -R_AB[2, 0]

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = sqrt((R_AB[0, 0]**2 + R_AB[1, 0]**2 +
                  R_AB[2, 1]**2 + R_AB[2, 2]**2)/2)

    y = arctan2(sin_y, cos_y)
    return z, y, x


def R_EL2n_E(R_EL):
    """
    Returns n-vector from the rotation matrix R_EL.

    Parameters
    ----------
    R_EL: 3 x 3 array
        Rotation matrix (direction cosine matrix) [no unit]

    Returns
    -------
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E.

    See also
    --------
    R_EN2n_E, n_E_and_wa2R_EL, n_E2R_EN
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = dot(R_EL, np.vstack((0, 0, -1)))
    return n_E


def R_EN2n_E(R_EN):
    """
    Returns n-vector from the rotation matrix R_EN.

    Parameters
    ----------
    R_EN: 3 x 3 array
        Rotation matrix (direction cosine matrix) [no unit]

    Returns
    -------
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E.

    See also
    --------
    n_E2R_EN, R_EL2n_E, n_E_and_wa2R_EL
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = dot(R_EN, np.vstack((0, 0, -1)))
    return n_E


def xyz2R(x, y, z):
    """
    Returns rotation matrix from 3 angles about new axes in the xyz-order.

    Parameters
    ----------
    x,y,z: real scalars
        Angles [rad] of rotation about new axes.

    Returns
    -------
    R_AB: 3 x 3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

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
    cz, sz = cos(z), sin(z)
    cy, sy = cos(y), sin(y)
    cx, sx = cos(x), sin(x)

    R_AB = array([[cy * cz, -cy * sz, sy],
                  [sy*sx*cz + cx*sz, -sy*sx*sz + cx*cz, -cy*sx],
                  [-sy*cx*cz + sx*sz, sy*cx*sz + sx*cz, cy*cx]])

    return np.squeeze(R_AB)


def zyx2R(z, y, x):
    """
    Returns rotation matrix from 3 angles about new axes in the zyx-order.

    Parameters
    ----------
    z, y, x: real scalars
        Angles [rad] of rotation about new axes.

    Returns
    -------
    R_AB: 3 x 3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

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
    cz, sz = cos(z), sin(z)
    cy, sy = cos(y), sin(y)
    cx, sx = cos(x), sin(x)

    R_AB = array([[cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy*cx],
                  [sz * cy,  cz * cx + sz * sy * sx, - cz * sx + sz*sy*cx],
                  [-sy, cy * sx, cy * cx]])

    return np.squeeze(R_AB)


class _GreatCircleDistance(object):
    __doc__ = """ Return great circle distance between two positions

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

    {}

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
    __doc__ = """Return Euclidean distance between two positions

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.
    radius: real scalar
        radius of sphere.

    Examples
    --------

    {}
    """.format(_examples.get_examples([5], OO=False))


@use_docstring_from(_EuclideanDistance)
def euclidean_distance(n_EA_E, n_EB_E, radius=6371009.0):
    d_AB = norm(n_EB_E - n_EA_E, axis=0) * radius
    return d_AB.ravel()


def n_EA_E_and_n_EB_E2azimuth(n_EA_E, n_EB_E, a=6378137, f=1.0/298.257223563,
                              R_Ee=None):
    """
    Return azimuth from A to B, relative to North:

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
    R_Ee : 2d array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    azimuth: n, array
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
    p_AB_N = dot(R_EN.T, p_AB_E)
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
    Return position B from azimuth and distance from position A

    Parameters
    ----------
    n_EA_E:  3 x n array
        n-vector(s) [no unit] of position A decomposed in E.
    distance_rad: n, array
        great circle distance [rad] from position A to B
    azimuth: n, array
        Angle [rad] the line makes with a meridian, taken clockwise from north.

    Returns
    -------
    n_EB_E:  3 x n array
        n-vector(s) [no unit] of position B decomposed in E.

    Examples
    --------

    {}

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
    Return the n-vector of the horizontal mean position.

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

    {}

    """.format(_examples.get_examples([7], OO=False))


@use_docstring_from(_MeanHorizontalPosition)
def mean_horizontal_position(n_EB_E):
    n_EM_E = unit(np.sum(n_EB_E, axis=1).reshape((3, 1)))
    return n_EM_E


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    print('Docstrings tested')


if __name__ == "__main__":
    test_docstrings()
