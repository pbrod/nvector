"""
Core geodesic functions
=======================
This file is part of NavLab and is available from www.navlab.net/nvector

"""
# pylint: disable=invalid-name

from __future__ import division, print_function
import warnings
import numpy as np
from numpy import arctan2, sin, cos, cross, dot, sqrt
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from nvector import _examples, license as _license
from nvector.rotation import E_rotation, n_E2R_EN, n_E2lat_lon  # @UnusedImport
from nvector.util import mdot, nthroot, unit, _nvector_check_length
from nvector._common import test_docstrings, use_docstring, _make_summary


__all__ = ['closest_point_on_great_circle',
           'cross_track_distance',
           'euclidean_distance',
           'great_circle_distance',
           'great_circle_distance_rad',
           'great_circle_normal',
           'interp_nvectors',
           'interpolate',
           'interp_nvectors',
           'intersect',
           'mean_horizontal_position',
           'lat_lon2n_E',
           'n_E2lat_lon',
           'n_EA_E_and_n_EB_E2p_AB_E',
           'n_EA_E_and_p_AB_E2n_EB_E',
           'n_EB_E2p_EB_E',
           'p_EB_E2n_EB_E',
           'n_EA_E_distance_and_azimuth2n_EB_E',
           'n_EA_E_and_n_EB_E2azimuth',
           'on_great_circle',
           'on_great_circle_path',
           ]


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
    # n_E = dot(R_Ee.T, nvec)
    n_E = np.matmul(R_Ee.T, nvec)
    return n_E


@use_docstring(_examples.get_examples_no_header([4], oo_solution=False))
def n_EB_E2p_EB_E(n_EB_E, depth=0, a=6378137, f=1.0 / 298.257223563, R_Ee=None):
    """
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
    {super}

    See also
    --------
    p_EB_E2n_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()

    n_EB_E = np.atleast_2d(n_EB_E)
    _nvector_check_length(n_EB_E)

    # Make sure to rotate the coordinates so that:
    # x -> north pole and yz-plane coincides with the equatorial
    # plane before using equation 22!
    n_EB_e = unit(np.matmul(R_Ee, n_EB_E))
    b = a * (1 - f)  # semi-minor axis

    # The following code implements equation (22) in Gade (2010):
    scale = np.vstack((1,
                       (1 - f),
                       (1 - f)))
    denominator = norm(n_EB_e / scale, axis=0, keepdims=True)

    # We first calculate the position at the origin of coordinate system L,
    # which has the same n-vector as B (n_EL_E = n_EB_E),
    # but lies at the surface of the Earth (z_EL = 0).

    p_EL_e = b / denominator * n_EB_e / scale**2
    # rotate back to the original coordinate system
    p_EB_E = np.matmul(R_Ee.T, p_EL_e - n_EB_e * depth)

    return p_EB_E


def _eccentricity2(f):
    """Returns the eccentricity squared"""
    return (2 - f) * f  # = 1-b**2/a**2


def _compute_k(a, e_2, q, Ryz_2):
    """returns the k value in equation (23) from Gade (2010)"""
    p = Ryz_2 / a ** 2
    r = (p + q - e_2 ** 2) / 6
    s = e_2 ** 2 * p * q / (4 * r ** 3)
    t = nthroot((1 + s + sqrt(s * (2 + s))), 3)
    # t = (1 + s + sqrt(s * (2 + s)))**(1. / 3)
    u = r * (1 + t + 1. / t)
    v = sqrt(u ** 2 + e_2 ** 2 * q)
    w = e_2 * (u + v - q) / (2 * v)
    return sqrt(u + v + w ** 2) - w


def _equation23(a, f, p_EB_E):
    """equation (23) from Gade (2010)"""
    Ryz_2 = p_EB_E[1, :]**2 + p_EB_E[2, :]**2
    Rx_2 = p_EB_E[0, :]**2
    e_2 = _eccentricity2(f)
    q = (1 - e_2) / (a ** 2) * Rx_2
    Ryz = sqrt(Ryz_2)  # Ryz = component of p_EB_E in the equatorial plane
    k = _compute_k(a, e_2, q, Ryz_2)
    d = k * Ryz / (k + e_2)
    temp0 = sqrt(d ** 2 + Rx_2)
    height = (k + e_2 - 1) / k * temp0  # Calculate height:
    x_scale = 1. / temp0
    yz_scale = x_scale * k / (k + e_2)
    return x_scale, yz_scale, -height


@use_docstring(_examples.get_examples_no_header([3], oo_solution=False))
def p_EB_E2n_EB_E(p_EB_E, a=6378137, f=1.0 / 298.257223563, R_Ee=None):
    """
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
    {super}

    See also
    --------
    n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E

    """
    if R_Ee is None:
        # R_Ee selects correct E-axes, see E_rotation for details
        R_Ee = E_rotation()

    # Make sure to rotate the coordinates so that:
    # x -> north pole and yz-plane coincides with the equatorial
    # plane before using equation 23!
    p_EB_e = np.matmul(R_Ee, p_EB_E)

    # The following code implements equation (23) from Gade (2010):
    x_scale, yz_scale, depth = _equation23(a, f, p_EB_e)

    n_EB_e_x = x_scale * p_EB_e[0, :]
    n_EB_e_y = yz_scale * p_EB_e[1, :]
    n_EB_e_z = yz_scale * p_EB_e[2, :]

    n_EB_e = np.vstack((n_EB_e_x, n_EB_e_y, n_EB_e_z))
    # Rotate back to the original coordinate system.
    n_EB_E = unit(np.matmul(R_Ee.T, n_EB_e))  # Ensure unit length

    return n_EB_E, depth


@use_docstring(_examples.get_examples_no_header([1], False))
def n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=6378137,
                             f=1.0 / 298.257223563, R_Ee=None):
    """
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
    {super}


    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_N, n_EB_E2p_EB_E,

    """

    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = n_EB_E2p_EB_E(n_EB_E, z_EB, a, f, R_Ee)
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


@use_docstring(_examples.get_examples_no_header([1], False))
def n_EA_E_and_n_EB_E2p_AB_N(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=6378137,
                             f=1.0 / 298.257223563, R_Ee=None):
    """
    Returns the delta vector from position A to B decomposed in N.

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
    p_AB_N:  3 x n array
        Cartesian position vector(s) from A to B, decomposed in N.

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
    {super}


    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E, n_EA_E_and_n_EB_E2p_AB_E

    """
    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA, z_EB, a, f, R_Ee)

    R_EN = n_E2R_EN(n_EA_E, R_Ee=R_Ee)

    # p_AB_N = dot(R_EN.T, p_AB_E)
    p_AB_N = mdot(np.swapaxes(R_EN, 1, 0), p_AB_E[:, None, ...]).reshape(3, -1)
    # (Note the transpose of R_EN: The "closest-rule" says that when
    # decomposing, the frame in the subscript of the rotation matrix that
    # is closest to the vector, should equal the frame where the vector is
    # decomposed. Thus the calculation np.dot(R_NE, p_AB_E) is correct,
    # since the vector is decomposed in E, and E is closest to the vector.
    # In the example we only had R_EN, and thus we must transpose it:
    # R_EN'=R_NE)
    return p_AB_N


@use_docstring(_examples.get_examples_no_header([2], oo_solution=False))
def n_EA_E_and_p_AB_E2n_EB_E(n_EA_E, p_AB_E, z_EA=0, a=6378137, f=1.0 / 298.257223563, R_Ee=None):
    """
    Returns position B from position A and delta vector decomposed in E.

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

    {super}

    See also
    --------
    n_EA_E_and_n_EB_E2p_AB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    n_EA_E, p_AB_E = np.atleast_2d(n_EA_E, p_AB_E)
    # Function 2. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = p_EA_E + p_AB_E
    n_EB_E, z_EB = p_EB_E2n_EB_E(p_EB_E, a, f, R_Ee)
    return n_EB_E, z_EB


@use_docstring(_examples.get_examples_no_header([2], oo_solution=False))
def n_EA_E_and_p_AB_N2n_EB_E(n_EA_E, p_AB_N, z_EA=0, a=6378137, f=1.0 / 298.257223563, R_Ee=None):
    """
    Returns position B from position A and delta vector decomposed in N.

    Parameters
    ----------
    n_EA_E:  3 x n array
        n-vector(s) [no unit] of position A, decomposed in E.
    p_AB_N:  3 x n array
        Cartesian position vector(s) from A to B, decomposed in N.
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
    A to position B (p_AB_N) are given. The output is the n-vector of position
    B (n_EB_E) and depth of B (z_EB).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    {super}

    See also
    --------
    n_EA_E_and_n_EB_E2p_AB_N,
    n_EA_E_and_p_AB_E2n_EB_E,
    n_E2R_EN
    """
    if R_Ee is None:
        R_Ee = E_rotation()
    n_EA_E, p_AB_N = np.atleast_2d(n_EA_E, p_AB_N)

    R_EN = n_E2R_EN(n_EA_E, R_Ee=R_Ee)

    # p_AB_E = dot(R_EN, p_AB_N)
    p_AB_E = mdot(R_EN, p_AB_N[:, None, ...]).reshape(3, -1)

    return n_EA_E_and_p_AB_E2n_EB_E(n_EA_E, p_AB_E, z_EA, a=a, f=f, R_Ee=R_Ee)


def _interp_vectors(t_i, t, nvectors, kind, window_length, polyorder, mode, cval):
    if window_length > 0:
        window_length = window_length + (window_length + 1) % 2  # make sure it is an odd integer
        options = dict(axis=1, mode=mode, cval=cval)
        normals = savgol_filter(nvectors, window_length, polyorder, **options)
    else:
        normals = nvectors

    normal_i = interp1d(t, normals, axis=1, kind=kind, bounds_error=False)(t_i)
    return normal_i.reshape(nvectors.shape[0], -1)


def interp_nvectors(t_i, t, nvectors, kind='linear', window_length=0, polyorder=2, mode='interp',
                    cval=0.0):
    """
    Returns interpolated values from nvector data.

    Parameters
    ----------
    t_i: real vector length m
        Vector of interpolation times.
    t: real vector length n
        Vector of times.
    nvectors: 3 x n array
        n-vector(s) [no unit] decomposed in E.
    kind: str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order) or as an
        integer specifying the order of the spline interpolator to use.
        Default is 'linear'.
    window_length: positive odd integer
        The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
        Default window_length=0, i.e. no smoothing.
    polyorder: int
        The order of the polynomial used to fit the samples.
        polyorder must be less than window_length.
    mode: 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
        Determines the type of extension to use for the padded signal to
        which the filter is applied.  When mode is 'constant', the padding
        value is given by cval.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree polyorder polynomial is fit to the
        last window_length values of the edges, and this polynomial is
        used to evaluate the last window_length // 2 output values.
    cval: scalar, optional
        Value to fill past the edges of the input if mode is 'constant'.
        Default is 0.0.

    Returns
    -------
    result: 3 x m array
        Interpolated n-vector(s) [no unit] decomposed in E.

    Notes
    -----
    The result for spherical Earth is returned.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import nvector as nv
    >>> lat, lon = nv.rad(np.arange(0, 10)), np.sin(nv.rad(np.linspace(-90, 70, 10)))
    >>> t = np.arange(len(lat))
    >>> t_i = np.linspace(0, t[-1], 100)
    >>> nvectors = nv.lat_lon2n_E(lat, lon)
    >>> nvectors_i = nv.interp_nvectors(t_i, t, nvectors, kind='cubic')
    >>> lati, loni = nv.deg(*nv.n_E2lat_lon(nvectors_i))
    >>> h = plt.plot(nv.deg(lon), nv.deg(lat), 'o', loni, lati, '-')
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    Interpolate noisy data
    >>> n = 50
    >>> lat = nv.rad(np.linspace(0, 9, n));
    >>> lon = np.sin(nv.rad(np.linspace(-90, 70, n))) + 0.05* np.random.randn(n)
    >>> t = np.arange(len(lat))
    >>> t_i = np.linspace(0, t[-1], 100)
    >>> nvectors = nv.lat_lon2n_E(lat, lon)
    >>> nvectors_i = nv.interp_nvectors(t_i, t, nvectors, 'cubic', 31)
    >>> [lati,loni] = nv.n_E2lat_lon(nvectors_i)
    >>> h = plt.plot(nv.deg(lon), nv.deg(lat), 'o', nv.deg(loni), nv.deg(lati), '-')
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()


    """
    normal_i = _interp_vectors(t_i, t, nvectors, kind, window_length, polyorder, mode, cval)

    return unit(normal_i, norm_zero_vector=np.nan)


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

    Notes
    -----
    The result for spherical Earth is returned.

    """

    n_EB_E_t0, n_EB_E_t1 = path
    n_EB_E_ti = unit(n_EB_E_t0 + ti * (n_EB_E_t1 - n_EB_E_t0),
                     norm_zero_vector=np.nan)
    return n_EB_E_ti


@use_docstring(_examples.get_examples_no_header([9], oo_solution=False))
def intersect(path_a, path_b):
    """
    Returns the intersection(s) between the great circles of the two paths

    Parameters
    ----------
    path_a, path_b: tuple of two n-vectors
        defining path A and path B, respectively.
        Path A and B has shape 2 x 3 x n and 2 x 3 x m, respectively.

    Returns
    -------
    n_EC_E : array of shape 3 x max(n, m)
        n-vector(s) [no unit] of position C decomposed in E.
        point(s) of intersection between paths.

    Notes
    -----
    The result for spherical Earth is returned.

    Examples
    --------
    {super}

    """
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


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def cross_track_distance(path, n_EB_E, method='greatcircle', radius=6371009.0):
    """
    Returns  cross track distance between path A and position B.

    Parameters
    ----------
    path: tuple of two n-vectors
        Two n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to measure the cross track distance to.
    method: string
        defining distance calculated. Options are: 'greatcircle' or 'euclidean'
    radius: real scalar
        radius of sphere. (default 6371009.0)

    Notes
    -----
    The result for spherical Earth is returned.

    Returns
    -------
    distance : array of length max(n, m)
        cross track distance(s)

    Examples
    --------
    {super}

    See also
    --------
    great_circle_normal,
    closest_point_on_great_circle,
    on_great_circle,
    on_great_circle_path
    """
    c_E = great_circle_normal(path[0], path[1])
    sin_theta = -np.sum(c_E * n_EB_E, axis=0)
    if method[0].lower() == 'e':
        return _euclidean_cross_track_distance(sin_theta, radius)
    return _great_circle_cross_track_distance(sin_theta, radius)


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def on_great_circle(path, n_EB_E, radius=6371009.0, atol=1e-8):
    """
    Returns True if position B is on great circle through path A.

    Parameters
    ----------
    path: tuple of two n-vectors
        Two n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to check to.
    radius: real scalar
        radius of sphere. (default 6371009.0)
    atol: real scalar
        The absolute tolerance parameter (See notes).

    Returns
    -------
    on : bool array of length max(n, m)
        True if position B is on great circle through path A.

    Notes
    -----
    The default value of `atol` is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity. However, `atol` should
    be carefully selected for the use case at hand. Typically the value
    should be set to the accepted error tolerance. For GPS data the error
    ranges from 0.01 m to 15 m.

    Examples
    --------
    {super}

    See also
    --------
    cross_track_distance
    """
    distance = np.abs(cross_track_distance(path, n_EB_E, radius=radius))
    return distance <= atol


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def on_great_circle_path(path, n_EB_E, radius=6371009.0, atol=1e-8):
    """
    Returns True if position B is on great circle and between endpoints of path A.

    Parameters
    ----------
    path: tuple of two n-vectors
        Two n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to measure the cross track distance to.
    radius: real scalar
        radius of sphere. (default 6371009.0)
    atol: real scalars
        The absolute tolerance parameter (See notes).

    Returns
    -------
    on : bool array of length max(n, m)
        True if position B is on great circle and between endpoints of path A.

    Notes
    -----
    The default value of `atol` is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity. However, `atol` should
    be carefully selected for the use case at hand. Typically the value
    should be set to the accepted error tolerance. For GPS data the error
    ranges from 0.01 m to 15 m.

    Examples
    --------
    {super}

    See also
    --------
    cross_track_distance, on_great_circle
    """
    n_EB_E, n_EA1_E, n_EA2_E = np.atleast_2d(n_EB_E, *path)
    scale = norm(n_EA2_E - n_EA1_E, axis=0)
    ti1 = norm(n_EB_E - n_EA1_E, axis=0) / scale
    ti2 = norm(n_EB_E - n_EA2_E, axis=0) / scale
    return (ti1 <= 1) & (ti2 <= 1) & on_great_circle(path, n_EB_E, radius, atol=atol)


@use_docstring(_examples.get_examples_no_header([10], oo_solution=False))
def closest_point_on_great_circle(path, n_EB_E):
    """
    Returns closest point C on great circle path A to position B.

    Parameters
    ----------
    path: tuple of 2 n-vectors of 3 x n arrays
        Two n-vectors of positions defining path A, decomposed in E.
    n_EB_E:  3 x m array
        n-vector(s) of position B to find the closest point to.

    Returns
    -------
    n_EC_E:  3 x max(m, n) array
        n-vector(s) of closest position C on great circle path A

    Examples
    --------
    {super}

    See also
    --------
    cross_track_distance, great_circle_normal

    """
    n_EA1_E, n_EA2_E = path
    c_E = great_circle_normal(n_EA1_E, n_EA2_E)

    c2 = cross(n_EB_E, c_E, axis=0)
    n_EC_E = unit(cross(c_E, c2, axis=0))
    return n_EC_E * np.sign(np.sum(n_EC_E * n_EB_E, axis=0, keepdims=True))


def great_circle_distance_rad(n_EA_E, n_EB_E):
    """
    Returns great circle distance in radians between positions A and B on a sphere

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.

    Notes
    -----
    The result for spherical Earth is returned.
    Formulae is given by equation (16) in Gade (2010) and is well
    conditioned for all angles.
    See also: https://en.wikipedia.org/wiki/Great-circle_distance.

    See also
    --------
    great_circle_distance
    """
    n_EA_E, n_EB_E = np.atleast_2d(n_EA_E, n_EB_E)
    sin_theta = norm(np.cross(n_EA_E, n_EB_E, axis=0), axis=0)
    cos_theta = np.sum(n_EA_E * n_EB_E, axis=0)
    # ill conditioned for small angles:
    # distance_rad_version1 = arccos(dot(n_EA_E,n_EB_E))

    # ill-conditioned for angles near pi/2 (and not valid above pi/2)
    # distance_rad_version2 = arcsin(norm(cross(n_EA_E,n_EB_E)))

    return np.arctan2(sin_theta, cos_theta)


@use_docstring(_examples.get_examples_no_header([5], oo_solution=False))
def great_circle_distance(n_EA_E, n_EB_E, radius=6371009.0):
    """
    Returns great circle distance between positions A and B on a sphere

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.
    radius: real scalar
        radius of sphere.

    Notes
    -----
    The result for spherical Earth is returned.
    Formulae is given by equation (16) in Gade (2010) and is well
    conditioned for all angles.
    See also: https://en.wikipedia.org/wiki/Great-circle_distance.

    Examples
    --------
    {super}

    See also
    --------
    great_circle_distance_rad
    """
    return great_circle_distance_rad(n_EA_E, n_EB_E) * radius


@use_docstring(_examples.get_examples_no_header([5], oo_solution=False))
def euclidean_distance(n_EA_E, n_EB_E, radius=6371009.0):
    """
    Returns Euclidean distance between positions A and B on a sphere

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, decomposed in E.
    radius: real scalar
        radius of sphere.

    Examples
    --------
    {super}
    """
    n_EB_E, n_EA_E = np.atleast_2d(n_EB_E, n_EA_E)
    d_AB = norm(n_EB_E - n_EA_E, axis=0).ravel() * radius
    return d_AB


def n_EA_E_and_n_EB_E2azimuth(n_EA_E, n_EB_E, a=6378137, f=1.0 / 298.257223563, R_Ee=None):
    """
    Returns azimuth from A to B, relative to North:

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, respectively, decomposed in E.
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

    See also
    --------
    great_circle_distance_rad, n_EA_E_distance_and_azimuth2n_EB_E
    """
    if R_Ee is None:
        R_Ee = E_rotation()

    #  Find p_AB_N (delta decomposed in N).
    p_AB_N = n_EA_E_and_n_EB_E2p_AB_N(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=a, f=f, R_Ee=R_Ee)

    # Find the direction (azimuth) to B, relative to north:

    return arctan2(p_AB_N[1], p_AB_N[0])


@use_docstring(_examples.get_examples_no_header([8], oo_solution=False))
def n_EA_E_distance_and_azimuth2n_EB_E(n_EA_E, distance_rad, azimuth, R_Ee=None):
    """
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

    Notes
    -----
    The result for spherical Earth is returned.

    Examples
    --------
    {super}

    See also
    --------
    n_EA_E_and_n_EB_E2azimuth, great_circle_distance_rad
    """

    if R_Ee is None:
        R_Ee = E_rotation()
    n_EA_E, distance_rad, azimuth = np.atleast_1d(n_EA_E, distance_rad, azimuth)
    # Step1: Find unit vectors for north and east:
    k_east_E = unit(cross(dot(R_Ee.T, [[1], [0], [0]]), n_EA_E, axis=0))
    k_north_E = cross(n_EA_E, k_east_E, axis=0)

    # Step2: Find the initial direction vector d_E:
    d_E = k_north_E * cos(azimuth) + k_east_E * sin(azimuth)

    # Step3: Find n_EB_E:
    n_EB_E = n_EA_E * cos(distance_rad) + d_E * sin(distance_rad)

#     north_pole = lat_lon2n_E(np.pi/2, 0, R_Ee)
#
#     start_on_pole = np.all(np.isclose(north_pole, n_EA_E), axis=0)
#
#     if np.any(start_on_pole):
#         n_EB_E[:, start_on_pole] = np.nan

    return n_EB_E


@use_docstring(_examples.get_examples_no_header([7], oo_solution=False))
def mean_horizontal_position(n_EB_E):
    """
    Returns the n-vector of the horizontal mean position.

    Parameters
    ----------
    n_EB_E:  3 x n array
        n-vectors [no unit] of positions Bi, decomposed in E.

    Returns
    -------
    p_EM_E:  3 x 1 array
        n-vector [no unit] of the mean positions of all Bi, decomposed in E.

    Notes
    -----
    The result for spherical Earth is returned.

    Examples
    --------
    {super}

    """
    n_EM_E = unit(np.sum(n_EB_E, axis=1).reshape((3, 1)))
    return n_EM_E


_odict = globals()
__doc__ = (__doc__  # @ReservedAssignment
           + _make_summary(dict((n, _odict[n]) for n in __all__))
           + 'License\n-------\n'
           + _license.__doc__)


if __name__ == "__main__":
    test_docstrings(__file__)
