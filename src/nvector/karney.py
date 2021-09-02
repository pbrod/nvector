'''
Created on 13. aug. 2021

References
----------
C. F. F. Karney, "Algorithms for geodesics",
J. Geodesy 87, 43-55 (2013);
https://doi.org/10.1007/s00190-012-0578-z
Addenda: https://geographiclib.sourceforge.io/geod-addenda.html

C. F. F. Karney, "Geodesics on an ellipsoid of revolution",


@author: pab
'''
import warnings
import numpy as np
from numpy import arctan2, sin, cos, tan, arctan, sqrt
# from scipy.special import ellipeinc, ellipkinc  # pylint: disable=no-name-in-module
from nvector.util import nthroot, eccentricity2, third_flattening, polar_radius


# A1 coefficients defined in eq. 17 in Karney:
A1_COEFFICIENTS = (1. / 256, 1. / 64., 1. / 4., 1.)
# C1 coefficients defined in eq. 18 in Karney:
C1_COEFFICIENTS = (
    (-1. / 32., 3. / 16., -1. / 2, ),  # C11
    (-9. / 2048., 1. / 32., -1. / 16.),  # C12
    (3. / 256, -1. / 48.),  # C13
    (3. / 512., -5. / 512.),  # C14
    (-7. / 1280.,),  # C15
    (-7. / 2048.,),  # C16
)
# CM1 coefficients defined in eq. 21 in Karney:
CM1_COEFFICIENTS = (
    (205. / 1536., -9. / 32., 1. / 2, ),  # CM11
    (1335. / 4096, -37. / 96., 5. / 16.),  # CM12
    (-75. / 128, 29. / 96.),  # CM13
    (-2391. / 2560., 539. / 1536.),  # CM14
    (3467. / 7680.,),  # CM15
    (38081. / 61440.,)  # CM16
)

# A2 coefficients defined in eq. 42 in Karney:
A2_COEFFICIENTS = (25. / 256., 9. / 64., 1./4., 1)
# C2 coefficients defined in eq. 43 in Karney:
C2_COEFFICIENTS = (
    (1. / 32., 1. / 16., 1./2.),  # C21
    (35. / 2048, 1./32., 3. / 16.),  # C22
    (5. / 256, 5. / 48.),  # C23
    (7. / 512., 35. / 512.),  # C24
    (63. / 1280.,),  # C25
    (77. / 2048.,),  # C26
)

# A3 coefficients defined in eq. 24 in Karney:
A3_COEFFICIENTS = (
    (-3. / 128.,),
    (-2. / 64., -3. / 64.),
    (-1. / 16., -3. / 16., -1. / 16.),
    (3. / 8., -1. / 8., -1. / 4.),
    (1. / 2., -1. / 2., ),
    (1., ))
# C3 coefficients defined in eq. 25 in Karney:
C3_COEFFICIENTS = (
    ((3, 128.), (2, 5, 128.), (-1, 3, 3, 64.), (-1, 0, 1, 8.), (-1, 1, 4.)),  # C_31
    ((5, 256.), (1, 3, 128.), (-3, -2, 3, 64.), (1, -3, 2, 32.)),  # C_32
    ((7, 512.), (-10, 9, 384.), (5, -9, 5, 192.)),  # C_33
    ((7, 512.), (-14, 7, 512.)),  # C_34
    ((21, 2560.),),  # C_35
)


def _astroid(x, y):
    """
    ASTROID  Solve the astroid equation

    K = ASTROID(X, Y) solves the quartic polynomial Eq. (55)

      K^4 + 2 * K^3 - (X^2 + Y^2 - 1) * K^2 - 2*Y^2 * K - Y^2 = 0

    for the positive root K.  X and Y are arrays of the same shape
    and the returned value K has the same shape.
    """
    x, y = np.atleast_1d(x, y)
    k = np.zeros(x.shape)
    p = x**2
    q = y**2
    r = (p + q - 1) / 6
    fl1 = ~((q == 0) & (r <= 0))
    p = p[fl1]
    q = q[fl1]
    r = r[fl1]
    S = p * q / 4
    r2 = r**2
    r3 = r * r2
    disc = S * (S + 2 * r3)
    u = r
    fl2 = disc >= 0
    T3 = S[fl2] + r3[fl2]
    T3 = T3 + (1 - 2 * (T3 < 0)) * sqrt(disc[fl2])
    T = nthroot(T3, 3)
    u[fl2] = u[fl2] + T + r2[fl2] / np.where(T != 0, T, np.inf)
    ang = arctan2(sqrt(-disc[~fl2]), -(S[~fl2] + r3[~fl2]))
    u[~fl2] = u[~fl2] + 2 * r[~fl2] * cos(ang / 3)
    v = sqrt(u**2 + q)
    uv = u + v
    fl2 = u < 0
    uv[fl2] = q[fl2] / (v[fl2] - u[fl2])
    w = (uv - q) / (2 * v)
    k[fl1] = uv / (sqrt(uv + w**2) + w)
    return k


def _eval_cij_coefs(coefficients, epsi, squared=True):
    epsi2 = epsi**2 if squared else epsi
    factor = 1.0
    c1x = []
    for coefs in coefficients:
        factor = factor * epsi
        c1x.append(factor*np.polyval(coefs, epsi2))
    return c1x


def _cosinesum(c, x, sine=True):
    """
    Returns the sum of the sine or cosine series using Clenshaw algorithm.

    Parameters
    ----------
    c : list
        sine or cosine series coefficients
    x : array-like
        argument to the sine or cosine series.
    sine: bool
        If True the sine series sum is returned otherwise the cosine series sum.

    Returns
    -------
    y : array-like
        If sine is True y = sum c[i-1] * sin( 2*i * x)
        otherwise y = sum c[i-1] * cos((2*i-1) * x) for i = 1, .... n
    """

    cosx, sinx = np.atleast_1d(cos(x), sin(x))
    n = len(c)
    ar = 2 * (cosx - sinx) * (cosx + sinx)
    y1 = np.zeros(sinx.shape)
    is_odd = n % 2
    if is_odd:
        y0 = c[-1]
        n = n - 1
    else:
        y0 = y1

    for k in range(n-1, -1, -2):
        y1 = ar * y0 - y1 + c[k]
        y0 = ar * y1 - y0 + c[k-1]

    if sine:
        return 2 * sinx * cosx * y0
    return cosx * (y0 - y1)


def _a3_coefs(n):
    """
    Returns the A3 coefficients defined in Eq. 24 in Karney evaluated at n.

    Parameters
    ----------
    n: real scalar
        third flattening of the ellipsoid
    """
    return [np.polyval(c, n) for c in A3_COEFFICIENTS]


def _c3_coefs(n):
    """
    Returns the C3 coefficients defined in Eq. 25 in Karney evaluated at n.

    Parameters
    ----------
    n: real scalar
        third flattening of the ellipsoid
    """
    return [[np.polyval(c[:-1], n) / c[-1] for c in coefs]
            for coefs in C3_COEFFICIENTS]


def _get_i3_fun(epsi, n=None, a3_coefs=None, c3_coefs=None):
    """
    Returns the I3 integral function defined in equation 8 in Karney

    Parameters
    ----------
    epsi: array-like
        normalized equatorial azimuth
    n: real scalar
        third flattening of the ellipsoid

    Returns
    -------
    i3fun : callable
        Integral function I3(sigma)

    Notes
    -----
    The I3 integral is defined as
      I3(sigma) = int (2-f)/(1+(1-f)*sqrt(1+k^2*sin(x)^2) dx from 0 to sigma

    Here
    f is the flattening
    n = f/(2-f) is the third flattening
    e = sqrt(f*(2-f)) is the eccentricity
    em = e/sqrt(1-e^2) is the second eccentricity
    alpha0 is the equatorial azimuth
    k = em*cos(alpha0)
    epsi = (sqrt(1+k^2)-1)/(sqrt(1+k^2)+1)
    sigma is the spherical arc length of the auxiliary sphere


    References
    ----------
    C. F. F. Karney, "Algorithms for geodesics",
    J. Geodesy 87, 43-55 (2013);
    https://doi.org/10.1007/s00190-012-0578-z
    Addenda: https://geographiclib.sourceforge.io/geod-addenda.html

    """
    if n is not None:
        a3_coefs = _a3_coefs(n)
        c3_coefs = _c3_coefs(n)

    a3 = np.polyval(a3_coefs, epsi)  # Eq. 24
    c3x = _eval_cij_coefs(c3_coefs, epsi, squared=False)  # Eq 25

    def i3fun(sigma):
        return a3*(sigma + _cosinesum(c3x, sigma, sine=True))
    return i3fun


def _get_i1_fun(epsi, return_inverse=True):
    """
    Returns the I1 integral function defined in equation 7 in Karney

    Parameters
    ----------
    epsi: array-like
        normalized equatorial azimuth

    Returns
    -------
    i1fun : callable
        Integral function I1(sigma)

    Notes
    -----
    The I1 integral is defined as
      I1(sigma) = int sqrt(1+k^2*sin(x)^2) dx from 0 to sigma

    Here
    f is the flattening

    e = sqrt(f*(2-f)) is the eccentricity
    em = e/sqrt(1-e^2) is the second eccentricity
    alpha0 is the equatorial azimuth
    k = em*cos(alpha0)
    epsi = (sqrt(1+k^2)-1)/(sqrt(1+k^2)+1) = k^2/(sqrt(1+k^2)+1)^2
    sigma is the spherical arc length of the auxiliary sphere


    References
    ----------
    C. F. F. Karney, "Algorithms for geodesics",
    J. Geodesy 87, 43-55 (2013);
    https://doi.org/10.1007/s00190-012-0578-z
    Addenda: https://geographiclib.sourceforge.io/geod-addenda.html

    """

    a1 = np.polyval(A1_COEFFICIENTS, epsi**2) / (1.0 - epsi)  # Eq 17
    c1x = _eval_cij_coefs(C1_COEFFICIENTS, epsi, squared=True)  # Eq 18

    def i1fun(sigma):
        """The I1 function"""
        return a1 * (sigma + _cosinesum(c1x, sigma, sine=True))

    if not return_inverse:
        return i1fun

    cm1x = _eval_cij_coefs(CM1_COEFFICIENTS, epsi, squared=True)  # Eq. 21

    def invi1fun(sdb):
        """The inverse of I1 function"""
        tau = sdb / a1
        return (tau + _cosinesum(cm1x, tau, sine=True))

#     k2 = -4 * epsi / (1-epsi)**2
#
#     def i1fun(sigma):
#         """The I1 function"""
#         return ellipeinc(sigma, k2)
#
#     def invi1fun(sdb, maxiter=20, atol=1e-12):
#         """The inverse of I1 function"""
#         sigma = sdb / a1
#
#         for _ in range(maxiter):
#             delta = (sdb - ellipeinc(sigma, k2)) / sqrt(1 - k2 * sin(sigma)**2)
#             sigma += delta
#             if np.all(np.abs(delta) < atol):
#                 break
#         return sigma

    return i1fun, invi1fun


def _get_jfun(epsi):
    epsi2 = epsi**2

    epsim1 = 1.0 - epsi
    a1 = np.polyval(A1_COEFFICIENTS, epsi2) / epsim1  # Eq 17
    a2 = np.polyval(A2_COEFFICIENTS, epsi2) * epsim1  # Eq 42
    a1m2 = a1-a2
    # Avoid subtraction of nearly equal numbers
#     a1m1 = np.polyval(A1_COEFFICIENTS[:-1], epsi2) * epsi2 / epsim1  # Eq 17
#     a2m1 = np.polyval(A2_COEFFICIENTS[:-1], epsi2) * epsi2 * epsim1  # Eq 42
#     a1m2 = epsi * (2.0 - epsi) / epsim1 + (a1m1 - a2m1)

    c1x = _eval_cij_coefs(C1_COEFFICIENTS, epsi, squared=True)  # Eq 18
    c2x = _eval_cij_coefs(C2_COEFFICIENTS, epsi, squared=True)  # Eq 43
    c1m2x = a1*np.array(c1x) - a2*np.array(c2x)

    def jfun(sigma):
        """The J function defined as I1(sigma)-I2(sigma)"""
        return a1m2 * sigma + _cosinesum(c1m2x, sigma, sine=True)

#     k2 = -4 * epsi / (1-epsi)**2
#
#     def jfun(sigma):
#         """The J function defined as I1(sigma) - I2(sigma)"""
#         return ellipeinc(sigma, k2) - ellipkinc(sigma, k2)

    return jfun


def truncate_small(x, small=0.06):
    """Truncate tiny values to zero"""
    y = np.where(np.abs(x) < small, small - (small - x), x)
    return np.where(x == 0, 0, y)


def normalize_angle(angle):
    """Normalize angle to range (-pi, pi]"""
    nangle = np.mod(angle+np.pi, 2*np.pi)-np.pi
    return np.where(nangle <= -np.pi, np.pi, nangle)


def _normalize_equatorial_azimuth(cos_alpha0, e2m):
    """
    Normalize the equatorial azimuth, alpha0, given the second eccentricity squared, e2m.
    """
    k2 = e2m * cos_alpha0 ** 2
    k1 = sqrt(1 + k2)
    epsi = k2 / (k1 + 1)**2  # Eq. 16
    return epsi


def _solve_triangle_NEA_direct(lat1, alpha1, f):
    """Returns alpha0, sigma1, w1, cos(alpha0), sin(alpha0)"""
    blat1 = arctan((1 - f) * tan(truncate_small(lat1)))  # Eq. 6
    return _solve_triangle_NEA(blat1, alpha1)


def _solve_triangle_NEA(blat1, alpha1):
    cos_alpha1, sin_alpha1 = cos(alpha1), sin(alpha1)
    cos_blat1, sin_blat1 = cos(blat1), sin(blat1)
    sin_alpha0 = sin_alpha1 * cos_blat1  # Eq. 5
    cos_alpha0 = np.abs(cos_alpha1 + 1j * sin_alpha1 * sin_blat1)
    # alpha0 = arctan2(sin_alpha0, cos_alpha0)  # Eq 10
    sigma1 = arctan2(sin_blat1, cos_alpha1 * cos_blat1)  # Eq 11
    w1 = arctan2(sin_alpha0 * sin(sigma1), cos(sigma1))  # Eq 12
    return sigma1, w1, cos_alpha0, sin_alpha0


def _solve_triangle_NEB_direct(sigma2, cos_alpha0, sin_alpha0):
    """Returns alpha2, blat2, w2"""
    cos_sigma2, sin_sigma2 = cos(sigma2), sin(sigma2)
    sin_blat2 = cos_alpha0 * sin_sigma2
    cos_blat2 = np.abs(cos_alpha0 * cos_sigma2 + 1j * sin_alpha0)
    w2 = arctan2(sin_alpha0 * sin_sigma2, cos_sigma2)  # Eq. 12
    blat2 = arctan2(sin_blat2, cos_blat2)  # Eq. 13
    alpha2 = arctan2(sin_alpha0, cos_alpha0 * cos_sigma2)  # Eq. 14
    return alpha2, blat2, w2


def _solve_triangle_NEB(cos_blat1, cos_blat2, sin_blat2, sin_alpha0, alpha1):
    cos_alpha2_cos_blat2 = np.sqrt(cos(alpha1)**2 * cos_blat1**2 + (cos_blat2**2 - cos_blat1**2))
    sin_alpha2_cos_blat2 = sin(alpha1)*cos_blat1
    alpha2 = arctan2(sin_alpha2_cos_blat2, cos_alpha2_cos_blat2)  # stable at both 0 and pi/2 angles
    # alpha20 = np.arccos(cos_alpha2_cos_blat2 / cos_blat2)  # Eq 45. in Karney
    sigma2 = arctan2(sin_blat2, cos_alpha2_cos_blat2)  # Eq 11
    w2 = arctan2(sin_alpha0 * sin(sigma2), cos(sigma2))  # Eq 12
    return sigma2, w2, alpha2


def sphere_distance_rad(lat1, lon1, lat2, lon2):
    """
    Returns surface distance between positions A and B as well as the azimuths.

    Parameters
    ----------
    lat1, lon1: real scalars or vectors of length m.
        latitude(s) and longitude(s) [rad] of position A.
    lat2, lon2: real scalars or vectors of length n.
        latitude(s) and longitude(s) [rad] of position B.

    Returns
    -------
    distance:  real scalars or vectors of length max(m,n).
        Surface distance [rad] from A to B on the sphere
    azimuth_a, azimuth_b: real scalars or vectors of length max(m,n).
        direction [rad] of line at position A and B relative to
        North, respectively.

    Notes
    -----
    Solves the inverse geodesic problem of finding the length and azimuths of the
    shortest geodesic between points specified by lat1, lon1, lat2, lon2 on a sphere.

    See also
    --------
    geodesic_distance

    """
    w = lon2 - lon1
    cos_b1, sin_b1 = cos(lat1), sin(lat1)
    cos_b2, sin_b2 = cos(lat2), sin(lat2)
    cos_w, sin_w = cos(w), sin(w)

    sin_a1 = cos_b2 * sin_w
    cos_a1 = cos_b1*sin_b2-sin_b1*cos_b2*cos_w
    sin_a2 = cos_b1 * sin_w
    cos_a2 = -cos_b2 * sin_b1 + sin_b2 * cos_b1 * cos_w
    cos_distance_rad = sin_b1*sin_b2+cos_b1*cos_b2*cos_w

    sin_distance_rad = np.hypot(sin_a1, cos_a1)

    azimuth_a = arctan2(sin_a1, cos_a1)  # Eq 49
    azimuth_b = arctan2(sin_a2, cos_a2)  # Eq 50
    distance_rad = arctan2(sin_distance_rad, cos_distance_rad)  # Eq 51
    return distance_rad, azimuth_a, azimuth_b


def geodesic_reckon(lat1, lon1, distance, azimuth, a=6378137, f=1.0 / 298.257223563):
    """
    Returns position B computed from position A, distance and azimuth.

    Parameters
    ----------
    lat1, lon1: real scalars or vectors of length k.
        latitude(s) and longitude(s) of position A.
    distance: real scalar or vector of length m.
        ellipsoidal distance [m] between position A and B.
    azimuth: real scalar or vector of length n.
        azimuth [rad] of line at position A.
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major half axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.

    Returns
    -------
    lat2, lon2:  arrays of length max(k,m,n).
        latitude(s) and longitude(s) of position A.
    azimuth_b: real scalars or vectors of length max(k,m,n).
        azimuth [rad] of line at position B.

    Examples
    --------
    >>> import numpy as np
    >>> import nvector as nv
    """

    alpha1 = truncate_small(azimuth)
    sigma1, w1, cos_alpha0, sin_alpha0 = _solve_triangle_NEA_direct(lat1, alpha1, f)

    # Determine sigma2:
    n = third_flattening(f)
    e2m = eccentricity2(f)[1]
    epsi = _normalize_equatorial_azimuth(cos_alpha0, e2m)

    i1fun, i1inv = _get_i1_fun(epsi)
    b = polar_radius(a, f)
    s1 = b * i1fun(sigma1)  # Eq. 7. I1(sigma1)
    s2 = s1 + distance
    sigma2 = i1inv(s2/b)  # Eq. 20: Inverse of I1 where I1 is defined in Eq. 7.

    alpha2, blat2, w2 = _solve_triangle_NEB_direct(sigma2, cos_alpha0, sin_alpha0)

    # Determine lamda12
    fun_i3 = _get_i3_fun(epsi, n)
    lamda1 = w1 - f * sin_alpha0 * fun_i3(sigma1)  # Eq. 8
    lamda2 = w2 - f * sin_alpha0 * fun_i3(sigma2)  # Eq. 8

    lamda12 = lamda2 - lamda1
    lon2 = lon1 + lamda12
    lat2 = arctan(tan(blat2)/(1-f))  # Eq. 6

    return lat2, lon2, alpha2


def _solve_alpha1(alpha1, blat1, blat2, true_lamda12, a, f, tol=1e-13):
    b = polar_radius(a, f)
    eta = third_flattening(f)
    e2, e2m = eccentricity2(f)

    sin_blat1, cos_blat1 = sin(blat1), cos(blat1)
    sin_blat2, cos_blat2 = sin(blat2), cos(blat2)

    def _newton_step(alpha1):
        """ See table 5 in Karney"""
        sigma1, w1, cos_alpha0, sin_alpha0 = _solve_triangle_NEA(blat1, alpha1)
        sigma2, w2, alpha2 = _solve_triangle_NEB(cos_blat1, cos_blat2, sin_blat2, sin_alpha0, alpha1)

        # Determine lamda12
        epsi = _normalize_equatorial_azimuth(cos_alpha0, e2m)
        fun_i3 = _get_i3_fun(epsi, eta)
        lamda1 = w1 - f * sin_alpha0 * fun_i3(sigma1)  # Eq. 8
        lamda2 = w2 - f * sin_alpha0 * fun_i3(sigma2)  # Eq. 8
        lamda12 = lamda2-lamda1

        # Update alpha1
        fun_j = _get_jfun(epsi)
        k2 = e2m * cos_alpha0 ** 2
        sin_sigma1, cos_sigma1 = sin(sigma1), cos(sigma1)
        sin_sigma2, cos_sigma2 = sin(sigma2), cos(sigma2)
        k_sin_s1 = sqrt(1+k2*sin_sigma1**2)
        k_sin_s2 = sqrt(1+k2*sin_sigma2**2)
        delta_j = fun_j(sigma2) - fun_j(sigma1)
        m12 = b*(k_sin_s2*cos_sigma1*sin_sigma2
                 - k_sin_s1*cos_sigma2*sin_sigma1
                 - cos_sigma1*cos_sigma2*delta_j)  # Eq 38
        # M12 = (cos_sigma1 * cos_sigma2
        #        + k_sin_s2 / k_sin_s1 * sin_sigma1 * sin_sigma2
        #        - sin_sigma1 * cos_sigma2 * delta_j / k_sin_s1)  # Eq 39
        cos_alpha2 = cos(alpha2)
        dlamda12_dalpha1 = np.where(np.abs(cos_alpha2) < tol,
                                    -sqrt(1 - e2 * cos_blat1**2) / sin_blat1 *
                                    (1 - np.sign(cos(alpha1))),
                                    m12 / a / (cos_alpha2 * cos_blat2))
        dlamda12 = true_lamda12-lamda12

        dalpha1 = dlamda12/dlamda12_dalpha1

        return dalpha1

    for i in range(20):
        dalpha1 = _newton_step(alpha1)
        alpha1 += dalpha1
        if np.all(np.abs(dalpha1) < 1e-12):
            break
    else:
        warnings.warn('Max iterations reached. Newton method did not converge.')
    return alpha1


def geodesic_distance(lat1, lon1, lat2, lon2, a=6378137, f=1.0 / 298.257223563):
    """
    Returns surface distance between positions A and B on an ellipsoid.

    Parameters
    ----------
    lat1, lon1: real scalars or vectors of length m.
        latitude(s) and longitude(s) of position A.
    lat2, lon2: real scalars or vectors of length n.
        latitude(s) and longitude(s) of position B.
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.
    R_Ee : 3 x 3 array
        rotation matrix defining the axes of the coordinate frame E.

    Returns
    -------
    distance:  real scalars or vectors of length max(m,n).
        Surface distance [m] from A to B on the ellipsoid
    azimuth_a, azimuth_b: real scalars or vectors of length max(m,n).
        direction [rad or deg] of line at position a and b relative to
        North, respectively.

    Notes
    -----
    Solves the inverse geodesic problem of finding the length and azimuths of the
    shortest geodesic between points specified by lat1, lon1, lat2, lon2 on an ellipsoid.

    See also
    --------
    sphere_distance_rad

    """
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(lat1, lon1, lat2, lon2)
    # assume lat1<=0 and lat1 < lat2 < -lat1
    b = polar_radius(a, f)
    eta = third_flattening(f)
    e2, e2m = eccentricity2(f)

    blat1 = arctan((1 - f) * tan(lat1))  # Eq 6
    blat2 = arctan((1 - f) * tan(lat2))  # Eq 6

    sin_blat1, cos_blat1 = sin(blat1), cos(blat1)
    sin_blat2, cos_blat2 = sin(blat2), cos(blat2)

    true_lamda12 = lon2 - lon1

    def vincenty():
        """See table 3 in Karney"""
        wbar = sqrt(1-e2*(0.5*(cos_blat1+cos_blat2))**2)  # Eq. 48
        w12 = true_lamda12/wbar
        # Determine sigma2:
        sigma12, alpha1, alpha2 = sphere_distance_rad(blat1, 0, blat2, w12)
        s12 = a*wbar*sigma12
        return s12, alpha1, alpha2, sigma12

    def _solve_astroid():
        """See table 4 in Karney"""
        delta = f * a * np.pi * cos_blat1**2
        x = (true_lamda12 - np.pi) * a * cos_blat1 / delta
        y = (blat1 + blat2) * a / delta
        mu = _astroid(x, y)
        alpha1 = np.where(y == 0,
                          arctan2(-x, sqrt(np.maximum(1 - x**2, 0))),
                          arctan2(-x / (1 + mu), y / mu))  # Eq. 56 and 57
        return alpha1

    def _solve_hybrid(alpha1):
        """See table 6 in Karney"""
        sigma1, w1, cos_alpha0, sin_alpha0 = _solve_triangle_NEA(blat1, alpha1)
        sigma2, w2, alpha2 = _solve_triangle_NEB(cos_blat1, cos_blat2, sin_blat2, sin_alpha0, alpha1)

        # Determine s12:
        epsi = _normalize_equatorial_azimuth(cos_alpha0, e2m)

        i1fun = _get_i1_fun(epsi, return_inverse=False)

        s12 = b * np.abs((i1fun(sigma2) - i1fun(sigma1)))  # Eq. 7. I1(sigma2)

        fun_i3 = _get_i3_fun(epsi, eta)
        lamda1 = w1 - f * sin_alpha0 * fun_i3(sigma1)  # Eq. 8
        lamda2 = w2 - f * sin_alpha0 * fun_i3(sigma2)  # Eq. 8
        lamda12 = lamda2-lamda1
        if np.any(np.abs(lamda12 - true_lamda12) > 1e-8):
            warnings.warn('Some positions did not converge using newton method!....')
        return s12, alpha2

    s12, alpha1, alpha2, sigma12 = vincenty()
    tol = 1e-12
    sin_lamda12 = sin(true_lamda12)
    meridional = np.abs(sin_lamda12) <= tol  # alpha1 = lamda12
    equatorial = (np.abs(lat1-lat2) <= tol) & (np.abs(lat1) <= tol)  # alpha1 = sign(lamda12)*pi
    nearly_antipodal = sigma12 > np.pi*(1-3*f*cos_blat1**2)
    alpha1 = np.where(nearly_antipodal, _solve_astroid(), alpha1)
    alpha1 = np.where(meridional, true_lamda12, alpha1)
    alpha1 = np.where(equatorial, np.sign(true_lamda12)*np.pi, alpha1)
    # alpha1 = np.deg2rad(161.914)
    short_distance = (s12 < a*1e-4)
    mask = ~(meridional | equatorial | short_distance)

    alpha1[mask] = _solve_alpha1(alpha1[mask], blat1[mask], blat2[mask], true_lamda12[mask], a, f)

    mask = ~short_distance

    s12[mask], alpha2[mask] = _solve_hybrid(alpha1[mask])
    return s12, alpha1, alpha2


if __name__ == '__main__':
    pass
