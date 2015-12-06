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
from __future__ import division
import numpy as np
import warnings


def deg(rad_angle):
    """deg Converts angle in radians to degrees.

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
    rad.
    """

    # Originated: 1996 Kenneth Gade, FFI

    deg_angle = rad_angle * 180 / np.pi
    return deg_angle


def rad(deg_angle):
    """
    rad Converts angle in degrees to radians.

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
    deg.
    """

    # Originated: 1996 Kenneth Gade, FFI

    rad_angle = deg_angle * np.pi / 180
    return rad_angle


def R_Ee():
    """
    R_Ee Selects axes of the coordinate frame E.

    This file controls the axes of the coordinate frame E (Earth-Centred,
    Earth-Fixed, ECEF) used by the other files in this library

    There are two choices of E-axes that are described in Table 2 in Gade
    (2010):

    * e: z-axis points to the North Pole and x-axis points to the point where
        latitude = longitude = 0. This choice is very common in many fields.

    * E: x-axis points to the North Pole,
         y-axis points towards longitude +90deg
        (east) and latitude = 0. This choice of axis directions ensures
        that at zero latitude and longitude, N (North-East-Down) has the
        same orientation as E. If roll/pitch/yaw are zero, also B (Body,
        forward, starboard, down) has this orientation. In this manner, the
        axes of E is chosen to correspond with the axes of N and B.

    Based on this we get:
    R_Ee=[0 0 1
          0 1 0
         -1 0 0]

    The above R_Ee should be returned from this file when using z-axis to the
    North pole (which is most common). When using x-axis to the North
    pole, R_Ee should be set to I (identity matrix) (since the files in
    this library are originally written for this option).

    Reference:
    ----------
    K Gade (2010): A Nonsingular Horizontal Position Representation,
    The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
    (www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf)
    """

    # Originated: 2015.02.19 Kenneth Gade and Kristian Svartveit, FFI
    # Modified:

    # Select axes of E (by commenting/uncommenting the two choices):
    # z-axis to the North Pole, most common choice:

    R_Ee = np.array([[0, 0, 1],
                     [0, 1, 0],
                     [-1, 0, 0]])

    # x-axis to the North Pole, corresponds to typical choices of N and B:
    # R_Ee_selected = np.eye(3)

    return R_Ee


def unit(vector):
    """
    unit Makes input vector unit length, i.e. norm==1.

    unit_vector = unit(vector)
    makes the general 1xm vector a unit_vector (norm==1).
    A matrix of vectors is accepted as input.
    """

    # Originated: 2002.07.04 Kenneth Gade, FFI
    # Modified:   2012.04.25 Kristian Svartveit, FFI: Vectorization
    # Modified:   2015.02.19 Kristian Svartveit, FFI: Bugfix and speedup
    # Modified:   2015.11.04 Kristian Svartveit, FFI: Fast for both matrix of
    #                        vectors and single vector input

    current_norm = np.linalg.norm(vector, axis=0)
    unit_vector = vector / current_norm
    idx = np.flatnonzero(current_norm == 0)
    unit_vector[:, idx] = 0
    unit_vector[0, idx] = 1
    return unit_vector


def lat_long2n_E(latitude, longitude):
    """
    lat_long2n_E Converts latitude and longitude to n-vector.
    n_E = lat_long2n_E(latitude,longitude)
    n-vector (n_E) is calculated from (geodetic) latitude and longitude.

    IN:
    latitude:  [rad]     Geodetic latitude
    longitude: [rad]

    OUT:
    n_E:       [no unit] n-vector decomposed in E (3x1 vector)

    The function also accepts vectors (1xm) with lat and long, then a 3xm
    matrix of n-vectors is returned.

    See also n_E2lat_long.
    """

    # Originated: 1999.02.23 Kenneth Gade, FFI
    # Modified:   2015.02.20 Kenneth Gade, FFI: Added possibility of using
    # alternative E axes

    # Equation (3) from Gade (2010):
    # R_Ee selects correct E-axes, see R_Ee.m for details
    nvec = np.vstack((np.sin(latitude),
                      np.sin(longitude) * np.cos(latitude),
                      -np.cos(longitude) * np.cos(latitude)))
    n_E = np.dot(R_Ee().T, nvec)
    return n_E


def n_E2lat_long(n_E):
    """
    n_E2lat_long Converts n-vector to latitude and lontitude.
    [latitude,longitude] = n_E2lat_long(n_E)
    Geodetic latitude and longitude are calculated from n-vector (n_E).

    IN:
    n_E:       [no unit] n-vector decomposed in E (3x1 vector)

    OUT:
    latitude:  [rad]     Geodetic latitude
    longitude: [rad]

    The function also accepts vectorized form (i.e. a 3xm matrix of n-vectors
    is input, returning 1xm vectors of latitude and longitude)

    See also lat_long2n_E.
    """

    # Originated: 1999.02.23 Kenneth Gade, FFI
    # Modified:   2004.11.23 Kenneth Gade, FFI: Accepts vectorized input
    # Modified:   2015.02.20 Kenneth Gade, FFI:
    #                        Added possibility of using alternative E axes

    check_length_deviation(n_E)

    n_E = np.dot(R_Ee(), n_E)
    # R_Ee selects correct E-axes, see R_Ee.m for details

    # CALCULATIONS:
    # Equation (5) in Gade (2010):
    longitude = np.arctan2(n_E[1, :], -n_E[2, :])

    # Equation (6) in Gade (2010) (Robust numerical solution)
    equatorial_component = np.sqrt(n_E[1, :]**2 + n_E[2, :]**2)
    # vector component in the equatorial plane
    latitude = np.arctan2(n_E[0, :], equatorial_component)
    # atan() could also be used since latitude is within [-pi/2,pi/2]

    # latitude=asin(n_E(1)) is a theoretical solution, but close to the Poles
    # it is ill-conditioned which may lead to numerical inaccuracies (and it
    # will give imaginary results for norm(n_E)>1)
    return latitude, longitude


def check_length_deviation(n_E):
    """
    n-vector should have length=1,  i.e. norm(n_E)=1.

    A deviation from 1 exceeding this limit gives a warning.
    This function only depends of the direction of n-vector, thus the warning
    is included only to give a notice in cases where a wrong input is given
    unintentionally (i.e. the input is not even approximately a unit vector).
    """
    length_deviation_warning_limit = 0.1
    length_deviation = np.abs(np.linalg.norm(n_E[:, 0]) - 1)
    # If a matrix of n-vectors is input,
    # only first is controlled to save time (assuming advanced users input
    # correct n-vectors)

    if length_deviation > length_deviation_warning_limit:
        warnings.warn('n-vector should have unit length: '
                      'norm(n_E)~=1 ! Error is: {}'.format(length_deviation))


def n_E2R_EN(n_E):
    """
    n_E2R_EN Finds the rotation matrix R_EN from n-vector.
    R_EN = n_E2R_EN(n_E) The rotation matrix (direction cosine matrix) R_EN
    is calculated based on n-vector (n_E).

    IN:
    n_E:   [no unit] n-vector decomposed in E (3x1 vector)

    OUT:
    R_EN:  [no unit] The resulting rotation matrix (direction cosine matrix)

    See also R_EN2n_E, n_E_and_wa2R_EL, R_EL2n_E.
    """

    # Originated: 2015.02.23 Kenneth Gade, FFI
    # Modified:

    check_length_deviation(n_E)
    n_E = unit(np.dot(R_Ee(), n_E))
    # Ensures unit length. R_Ee selects correct E-axes, see R_Ee.m for details.
    # Note: In code where the norm of the input n_EB_E is guaranteed to be 1,
    # the use of the unit-function can be removed, to gain some speed.

    # CALCULATIONS:
    # N coordinate frame (North-East-Down) is defined in Table 2 in Gade (2010)
    # Find z-axis of N (Nz):
    Nz_E = -n_E  # z-axis of N (down) points opposite to n-vector

    # Find y-axis of N (East)(remember that N is singular at Poles)
    # Equation (9) in Gade (2010):
    # Ny points perpendicular to the plane
    Ny_E_direction = np.cross([[1], [0], [0]], n_E, axis=0)
    # formed by n-vector and Earth's spin axis
    outside_poles = (np.linalg.norm(Ny_E_direction) != 0)
    if outside_poles:
        Ny_E = unit(Ny_E_direction)
    else:  # Pole position:
        Ny_E = np.array([[0], [1], [0]])  # selected y-axis direction

    # Find x-axis of N (North):
    Nx_E = np.cross(Ny_E, Nz_E, axis=0)  # Final axis found by right hand rule

    # Form R_EN from the unit vectors:
    R_EN = np.dot(R_Ee().T, np.hstack((Nx_E, Ny_E, Nz_E)))
    # R_Ee selects correct E-axes, see R_Ee.m for details

    return R_EN


def n_E_and_wa2R_EL(n_E, wander_azimuth):
    """
    n_E_and_wa2R_EL Finds R_EL from n-vector and wander azimuth angle.
    R_EL = n_E_and_wa2R_EL(n_E,wander_azimuth) Calculates the rotation matrix
    (direction cosine matrix) R_EL using n-vector (n_E) and the wander
    azimuth angle.
    When wander_azimuth=0, we have that N=L (See Table 2 in Gade (2010) for
    details)

    IN:
    n_E:        [no unit] n-vector decomposed in E (3x1 vector)
    wander_azimuth: [rad] The angle between L's x-axis and north, pos about
    L's z-axis

    OUT:
    R_EL:       [no unit] The resulting rotation matrix (3x3)

    See also R_EL2n_E, R_EN2n_E, n_E2R_EN.
    """

    # Originated: 1999.02.23 Kenneth Gade, FFI
    # Modified:   2015.02.20 Kenneth Gade, FFI: Added possibility of using
    # alternative E axes

    latitude, longitude = n_E2lat_long(n_E)

    # Reference: See start of Section 5.2 in Gade (2010):
    R_EL = np.dot(R_Ee().T, xyz2R(longitude, -latitude, wander_azimuth))
    # R_Ee selects correct E-axes, see R_Ee.m for details
    return R_EL


def check_backward_compatibility(a, f):
    """
    Previously, custom ellipsoid was spesified by a and b in this function.
    However, for more spherical globes than the Earth, or if f has more
    decimals than in WGS-84, using f and a as input will give better
    numerical precicion than a and b.

    old input number 3, 4: Polar_semi_axis (b), equatorial_semi_axis (a)
    """
    if f > 1e6:  # Checks if a is given as f (=old input)
        warnings.warn('Deprecated call: '
                      'Polar_semi_axis (b), equatorial_semi_axis (a) '
                      'Use a=equatorial radius, f=flattening of ellipsoid')
        f_new = 1 - a / f
        a = f
        f = f_new  # switch old inputs to new format
    return a, f


def n_EB_E2p_EB_E(n_EB_E, z_EB=None, a=6378137, f=1 / 298.257223563):
    """
    n_EB_E2p_EB_E Converts n-vector to Cartesian position vector in meters.

    p_EB_E = n_EB_E2p_EB_E(n_EB_E)

    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, n_EB_E. The function converts
    to cartesian position vector ("ECEF-vector"), p_EB_E, in meters.
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also nonsingular as both n-vector and p-vector are nonsingular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    p_EB_E = n_EB_E2p_EB_E(n_EB_E,z_EB)

    Depth of B, z_EB, is also specified,
    z_EB = 0 is used when not spefified.

    p_EB_E = n_EB_E2p_EB_E(n_EB_E,z_EB,a)
    Spherical Earth with radius a is used in stead of WGS-84.

    p_EB_E = n_EB_E2p_EB_E(n_EB_E,z_EB,a,f)
    Ellipsoidal Earth model with semi-major axis a and flattening f is used
    in stead of WGS-84.

    IN:
    n_EB_E:  [no unit] n-vector of position B, decomposed in E (3x1 vector).
    z_EB:    [m]     (Optional, assumed to be zero if not given) Depth of system B,
                     relative to the ellipsoid (z_EB = -height)
    a:       [m]       (Optional) Semi-major axis of the Earth ellipsoid

    f:       [no unit] (Optional) Flattening of the Earth ellipsoid

    OUT:
    p_EB_E:  [m]       Cartesian position vector from E to B, decomposed in E (3x1 vector).

    The function also accepts vectorized form, i.e. n_EB_E is a 3xn matrix, z_EB is
    a 1xn vector and p_EB_E is a 3xn matrix.

    See also p_EB_E2n_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
    """

    # Originated: 2004.11.17 Kenneth Gade and Brita Hafskjold, FFI
    # Modified:   2015.02.20 Kenneth Gade, FFI: Added possibility of using
    # alternative E axes

    check_length_deviation(n_EB_E)

    n_EB_E = unit(np.dot(R_Ee(), n_EB_E))
    # Ensures unit length. R_Ee selects correct E-axes, see R_Ee function for
    # details.
    # Note: In code where the norm of the input n_EB_E is guaranteed to be 1,
    # the use of the unit-function can be removed, to gain some speed.
    if z_EB is None:
        z_EB = np.zeros((1, np.shape(n_EB_E)[1]))

    a, f = check_backward_compatibility(a, f)
    # WGS-84 ellipsoid is used
    # a = 6378137  the equatorial radius of the Earth-ellipsoid
    # f = 1/298.257223563 the flattening of the Earth-ellipsoid

    # CALCULATIONS:

    # semi-minor axis:
    b = a * (1 - f)

    # The following code implements equation (22) in Gade (2010):

    scale = np.vstack((1,
                       (1 - f),
                       (1 - f)))
    denominator = np.linalg.norm(n_EB_E / scale, axis=0)

    # We first calculate the position at the origin of coordinate system L,
    # which has the same n-vector as B (n_EL_E = n_EB_E),
    # but lies at the surface of the Earth (z_EL = 0).

    p_EL_E = b / denominator * n_EB_E / scale**2
    p_EB_E = np.dot(R_Ee().T, p_EL_E - n_EB_E * z_EB)

    return p_EB_E


def p_EB_E2n_EB_E(p_EB_E, a=6378137, f=1 / 298.257223563):
    """
     p_EB_E2n_EB_E  Converts Cartesian position vector in meters to n-vector.
    [n_EB_E,z_EB] = p_EB_E2n_EB_E(p_EB_E)
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as cartesian position vector p_EB_E, in meters.
    ("ECEF-vector"). The function converts to n-vector, n_EB_E and its
    depth, z_EB.
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also nonsingular as both n-vector and p-vector are nonsingular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    [n_EB_E,z_EB] = p_EB_E2n_EB_E(p_EB_E,a) Spherical Earth with radius a is
    used in stead of WGS-84.

    [n_EB_E,z_EB] = p_EB_E2n_EB_E(p_EB_E,a,f) Ellipsoidal Earth model with
    semi-major axis a and flattening f is used in stead of WGS-84.

    IN:
    p_EB_E: [m]       Cartesian position vector from E to B, decomposed in E (3x1 vector).
    a:      [m]       (Optional) Semi-major axis of the Earth ellipsoid
    f:      [no unit] (Optional) Flattening of the Earth ellipsoid

    OUT:
    n_EB_E: [no unit] n-vector  representation of position B, decomposed in E (3x1 vector).
    z_EB:   [m]       Depth of system B relative to the ellipsoid (z_EB = -height).


    The function also accepts vectorized form, i.e. p_EB_E is a 3xn matrix,
    n_EB_E is a 3xn matrix and z_EB is a 1xn vector.

    See also n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
    """

    # Originated: 2004.11.17 Kenneth Gade and Brita Hafskjold, FFI
    # Modified:   2007.03.02 Brita Hafskjold Gade, FFI
    #             Replaced formulas to get full numerical accuracy at all positions:
    # Modified:   2014.08.22 Kenneth Gade, FFI:
    #                Added possibility of vectorized input/output

    # INPUT HANDLING:

    p_EB_E = np.dot(R_Ee(), p_EB_E)
    # R_Ee selects correct E-axes, see R_Ee.m for details
    a, f = check_backward_compatibility(a, f)

    # e_2 = eccentricity**2
    e_2 = 2 * f - f**2  # = 1-b^2/a^2;

    # The following code implements equation (23) from Gade (2010):
    R_2 = p_EB_E[1, :]**2 + p_EB_E[2, :]**2
    R = np.sqrt(R_2)   # R = component of p_EB_E in the equatorial plane

    p = R_2 / a**2
    q = (1 - e_2) / (a**2) * p_EB_E[0, :]**2
    r = (p + q - e_2**2) / 6

    s = e_2**2 * p * q / (4 * r**3)
    # t = nthroot((1 + s + sqrt(s.*(2+s))), 3);
    t = (1 + s + np.sqrt(s * (2 + s)))**(1. / 3)
    u = r * (1 + t + 1. / t)
    v = np.sqrt(u**2 + e_2**2 * q)

    w = e_2 * (u + v - q) / (2 * v)
    k = np.sqrt(u + v + w**2) - w
    d = k * R / (k + e_2)

    # Calculate height:
    height = (k + e_2 - 1) / k * np.sqrt(d**2 + p_EB_E[0, :]**2)

    temp = 1. / np.sqrt(d**2 + p_EB_E[0, :]**2)

    n_EB_E_x = temp * p_EB_E[0, :]
    n_EB_E_y = temp * k / (k + e_2) * p_EB_E[1, :]
    n_EB_E_z = temp * k / (k + e_2) * p_EB_E[2, :]

    n_EB_E = np.vstack((n_EB_E_x, n_EB_E_y, n_EB_E_z))

    # Ensure unit length:
    n_EB_E = unit(np.dot(R_Ee().T, n_EB_E))

    z_EB = -height
    return n_EB_E, z_EB


def n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA=None, z_EB=None,
                             a=6378137, f=1 / 298.257223563):
    """
    n_EA_E_and_n_EB_E2p_AB_E From two positions A and B, finds the delta
    position.

    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E,n_EB_E)
    The n-vectors for positions A (n_EA_E) and B (n_EB_E) are given. The
    output is the delta vector from A to B (p_AB_E).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also nonsingular as both n-vector and p-vector are nonsingular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids (or spheres)
    might be specified.

    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E,n_EB_E,z_EA)
    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E,n_EB_E,z_EA,z_EB)
    Depth(s) of A, z_EA (and of B, z_EB) are also specified, z_EA = 0 (and z_EB = 0)
    is used when not spefified.

    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E,n_EB_E,z_EA,z_EB,a)
    Spherical Earth with radius a is used in stead of WGS-84.

    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E,n_EB_E,z_EA,z_EB,a,f)
    Ellipsoidal Earth model with semi-major axis a and flattening f is used
    in stead of WGS-84.

    IN:
    n_EA_E:  [no unit] n-vector of position A, decomposed in E (3x1 vector).
    n_EB_E:  [no unit] n-vector of position B, decomposed in E (3x1 vector).
    z_EA:    [m]       (Optional, assumed to be zero if not given) Depth of system A,
                       relative to the ellipsoid (z_EA = -height).
    z_EB:    [m]       (Optional, assumed to be zero if not given) Depth of system B,
                       relative to the ellipsoid (z_EB = -height).
    a:       [m]       (Optional) Semi-major axis of the Earth ellipsoid
    f:       [no unit] (Optional) Flattening of the Earth ellipsoid

    OUT:
    p_AB_E:  [m]       Position vector from A to B, decomposed in E (3x1 vector).

    The function also accepts vectorized form, i.e. n_EA_E and n_EB_E are 3xn matrixes,
    z_EA and z_EB are 1xn vectors and p_AB_E is a 3xn matrix.

    See also n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E.
    """

    # Originated: 2004.07.07 Kenneth Gade, FFI
    # Modified:

    # The optional inputs a and f are forwarded to the kernel function (which
    # uses the same syntax):
    n = np.shape(n_EA_E)[1]
    if z_EA is None:
        z_EA = np.zeros((1, n))
    if z_EB is None:
        z_EB = np.zeros((1, n))

    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f)
    p_EB_E = n_EB_E2p_EB_E(n_EB_E, z_EB, a, f)
    p_AB_E = -p_EA_E + p_EB_E
    return p_AB_E


def n_EA_E_and_p_AB_E2n_EB_E(n_EA_E,p_AB_E,z_EA=None, a=6378137,
                             f=1 / 298.257223563):
    """
     n_EA_E_and_p_AB_E2n_EB_E From position A and delta, finds position B.
     n_EB_E      = n_EA_E_and_p_AB_E2n_EB_E(n_EA_E,p_AB_E)
    [n_EB_E,z_EB] = n_EA_E_and_p_AB_E2n_EB_E(n_EA_E,p_AB_E)
    The n-vector for position A (n_EA_E) and the position-vector from position
    A to position B (p_AB_E) are given. The output is the n-vector of position
    B (n_EB_E) and depth of B (z_EB).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also nonsingular as both n-vector and p-vector are nonsingular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids (or spheres)
    might be specified.

    [n_EB_E,z_EB] = n_EA_E_and_p_AB_E2n_EB_E(n_EA_E,p_AB_E,z_EA) Depth of A, z_EA,
    is also specified, z_EA = 0 is used when not spefified.

    [n_EB_E,z_EB] = n_EA_E_and_p_AB_E2n_EB_E(n_EA_E,p_AB_E,z_EA,a)
    Spherical Earth with radius a is used in stead of WGS-84.

    [n_EB_E,z_EB] = n_EA_E_and_p_AB_E2n_EB_E(n_EA_E,p_AB_E,z_EA,a,f)
    Ellipsoidal Earth model with semi-major axis a and flattening f is used
    in stead of WGS-84.

    IN:
    n_EA_E:  [no unit] n-vector of position A, decomposed in E (3x1 vector).
    p_AB_E:  [m]       Position vector from A to B, decomposed in E (3x1 vector).
    z_EA:    [m]       (Optional, assumed to be zero if not given) Depth of system A,
                       relative to the ellipsoid (z_EA = -height).
    a:       [m]       (Optional) Semi-major axis of the Earth ellipsoid
    f:       [no unit] (Optional) Flattening of the Earth ellipsoid

    OUT:
    n_EB_E:  [no unit] n-vector of position B, decomposed in E (3x1 vector).
    z_EB:    [m]       Depth of system B, relative to the ellipsoid (z_EB = -height).

    The function also accepts vectorized form, i.e. n_EA_E and p_AB_E are 3xn matrixes,
    z_EA and z_EB are 1xn vectors and n_EB_E is a 3xn matrix.

    See also n_EA_E_and_n_EB_E2p_AB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E.
    """

    # Originated: 2004.07.07 Kenneth Gade, FFI
    # Modified:
    a, f = check_backward_compatibility(a, f)

    ## CALCULATIONS:
    # Function 2. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f)
    p_EB_E = p_EA_E + p_AB_E
    n_EB_E, z_EB = p_EB_E2n_EB_E(p_EB_E, a, f)
    return n_EB_E, z_EB

def R2xyz(R_AB):
    """
    Three angles about new axes in the xyz order are found from a rotation matrix.

    [x,y,z] = R2xyz(R_AB) 3 angles x,y,z about new axes (intrinsic) in the
    order x-y-z are found from the rotation matrix R_AB. The angles (called
    Euler angles or Tait-Bryan angles) are defined by the following procedure
    of successive rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle x about its x-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle z about its NEWEST z-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    IN:
    R_AB  [no unit]    3x3 rotation matrix (direction cosine matrix) such that
                    the relation between a vector v decomposed in A and B is
                    given by: v_A = R_AB * v_B

    OUT:
    x,y,z [rad]        Angles of rotation about new axes.

    See also xyz2R, R2zyx, xyz2R.
    """

    # Originated: 1996.10.01 Kenneth Gade, FFI
    # Modified:

    # atan2: [-pi pi]
    z = np.arctan2(-R_AB[0, 1], R_AB[0, 0])
    x = np.arctan2(-R_AB[1, 2], R_AB[2, 2])

    sin_y = R_AB[0, 2]

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = np.sqrt((R_AB[0, 0]**2 + R_AB[0, 1]**2 +
                     R_AB[1, 2]**2 + R_AB[2, 2]**2)/2)

    y = np.arctan2(sin_y, cos_y)
    return x, y, z


def R2zyx(R_AB):
    """
    Three angles about new axes in the zyx order are found from a rotation matrix.

    [z,y,x] = R2zyx(R_AB) 3 angles z,y,x about new axes (intrinsic) in the
    order z-y-x are found from the rotation matrix R_AB. The angles (called
    Euler angles or Tait-Bryan angles) are defined by the following procedure
    of successive rotations:
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

    IN:
    R_AB  [no unit]    3x3 rotation matrix (direction cosine matrix) such that the
                    relation between a vector v decomposed in A and B is
                    given by: v_A = R_AB * v_B

    OUT:
    z,y,x [rad]        Angles of rotation about new axes.

    See also zyx2R, xyz2R, R2xyz.
    """

    # Originated: 1996.10.01 Kenneth Gade, FFI
    # Modified:

    # atan2: [-pi pi]
    z = np.arctan2(R_AB[1, 0], R_AB[0, 0])
    x = np.arctan2(R_AB[2, 1], R_AB[2, 2])

    sin_y = -R_AB[2, 0]

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = np.sqrt((R_AB[0, 0]**2+R_AB[1, 0]**2 +
                     R_AB[2, 1]**2+R_AB[2, 2]**2)/2)

    y = np.arctan2(sin_y, cos_y)
    return z, y, x


def R_EL2n_E(R_EL):
    """
    Finds n-vector from R_EL.

    n_E = R_EL2n_E(R_EL) n-vector is found from the rotation matrix
    (direction cosine matrix) R_EL.

    IN:
    R_EL:  [no unit] Rotation matrix (direction cosine matrix)

    OUT:
    n_E:   [no unit] n-vector decomposed in E (3x1 vector)

    See also R_EN2n_E, n_E_and_wa2R_EL, n_E2R_EN.
    """

    # Originated: 1999.02.23 Kenneth Gade, FFI
    # Modified:

    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = np.dot(R_EL, np.r_[0, 0, -1].T)
    return n_E


def R_EN2n_E(R_EN):
    """
    Finds n-vector from R_EN.

    n_E = R_EN2n_E(R_EN)
    n-vector is found from the rotation matrix (direction cosine matrix)
    R_EN.

    IN:
    R_EN:  [no unit] Rotation matrix (direction cosine matrix)

    OUT:
    n_E:   [no unit] n-vector decomposed in E (3x1 vector)

    See also n_E2R_EN, R_EL2n_E, n_E_and_wa2R_EL.
    """

    # Originated: 1999.02.23 Kenneth Gade, FFI
    # Modified:

    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = np.dot(R_EN, np.r_[0, 0, -1].T);
    return n_E


def xyz2R(x, y, z):
    """
    xyz2R Creates a rotation matrix from 3 angles about new axes in the xyz order.
    R_AB = xyz2R(x,y,z) The rotation matrix R_AB is created based on 3 angles
    x,y,z about new axes (intrinsic) in the order x-y-z. The angles (called
    Euler angles or Tait-Bryan angles) are defined by the following procedure
    of successive rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle x about its x-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle z about its NEWEST z-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    IN:
    x,y,z [rad]        Angles of rotation about new axes.

    OUT:
    R_AB  [no unit]    3x3 rotation matrix (direction cosine matrix) such that the
                    relation between a vector v decomposed in A and B is
                    given by: v_A = R_AB * v_B

    See also R2xyz, zyx2R, R2zyx.
    """

    # Originated: 1996.10.01 Kenneth Gade, FFI
    # Modified:

    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)

    R_AB = np.array([[cy * cz, -cy * sz, sy],
                     [sy*sx*cz + cx*sz, -sy*sx*sz + cx*cz, -cy*sx],
                     [-sy*cx*cz + sx*sz, sy*cx*sz + sx*cz, cy*cx]])

    return R_AB


def zyx2R(z,y,x):
    """
    Creates a rotation matrix from 3 angles about new axes in the zyx order.
    R_AB = zyx2R(z,y,x) The rotation matrix R_AB is created based on 3 angles
    z,y,x about new axes (intrinsic) in the order z-y-x. The angles (called
    Euler angles or Tait-Bryan angles) are defined by the following procedure
    of successive rotations:
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

    IN:
    z,y,x [rad]        Angles of rotation about new axes.

    OUT:
    R_AB  [no unit]    3x3 rotation matrix (direction cosine matrix) such that the
                    relation between a vector v decomposed in A and B is
                    given by: v_A = R_AB * v_B

    See also R2zyx, xyz2R, R2xyz.
    """
    # Originated: 1996.10.01 Kenneth Gade, FFI
    # Modified:

    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)

    R_AB = np.array([[cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy*cx],
                     [sz * cy,  cz * cx + sz * sy * sx, - cz * sx + sz * sy*cx],
                     [-sy, cy * sx, cy * cx]])

    return R_AB


if __name__ == '__main__':
    pass
