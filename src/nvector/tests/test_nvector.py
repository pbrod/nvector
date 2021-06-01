"""
This file contains solutions to the examples given at
www.navlab.net/nvector

The content of this file is based on the following publication:

Gade, K. (2010). A Nonsingular Horizontal Position Representation, The Journal
of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
(www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf)

Copyright (c) 2015, Norwegian Defence Research Establishment (FFI)
All rights reserved.

Originated: 2015.03.26 Kenneth Gade, FFI

NOTES:
- All angles are by default assumed to be in radians, if an angle is
in degrees, the variable name has the following ending: _deg

- The dot product (inner product) of vectors x and y is written dot(x,y).
"""
from functools import partial
import numpy as np

from nvector.util import unit, deg, rad
from nvector.rotation import zyx2R, n_E2R_EN
from nvector.core import (lat_lon2n_E, n_E2lat_lon,
                          n_EA_E_and_n_EB_E2p_AB_E,
                          n_EA_E_and_p_AB_E2n_EB_E,
                          p_EB_E2n_EB_E,
                          n_EB_E2p_EB_E,
                          mean_horizontal_position,
                          great_circle_distance,
                          euclidean_distance,
                          cross_track_distance,
                          closest_point_on_great_circle,
                          n_EA_E_distance_and_azimuth2n_EB_E,
                          n_EA_E_and_n_EB_E2azimuth)
from numpy.testing import assert_allclose as _assert_allclose  # @UnresolvedImport

assert_allclose = partial(_assert_allclose, atol=1e-15)


def test_Ex1_A_and_B_to_delta_in_frame_N():

    # Positions A and B are given in (decimal) degrees and depths:
    lat_EA, lon_EA, z_EA = rad(1), rad(2), 3
    lat_EB, lon_EB, z_EB = rad(4), rad(5), 6

    # Find the exact vector between the two positions, given in meters
    # north, east, and down, i.e. find p_AB_N.

    # SOLUTION:
    # Step1: Convert to n-vectors (rad() converts to radians):
    n_EA_E = lat_lon2n_E(lat_EA, lon_EA)
    n_EB_E = lat_lon2n_E(lat_EB, lon_EB)

    # Step2: Find p_AB_E (delta decomposed in E).
    # WGS-84 ellipsoid is default:
    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA, z_EB)

    # Step3: Find R_EN for position A:
    R_EN = n_E2R_EN(n_EA_E)

    # Step4: Find p_AB_N
    p_AB_N = np.dot(R_EN.T, p_AB_E)
    # (Note the transpose of R_EN: The "closest-rule" says that when
    # decomposing, the frame in the subscript of the rotation matrix that
    # is closest to the vector, should equal the frame where the vector is
    # decomposed. Thus the calculation np.dot(R_NE, p_AB_E) is correct,
    # since the vector is decomposed in E, and E is closest to the vector.
    # In the example we only had R_EN, and thus we must transpose it:
    # R_EN'=R_NE)

    # Step5: Also find the direction (azimuth) to B, relative to north:
    azimuth = np.arctan2(p_AB_N[1], p_AB_N[0])
    # positive angle about down-axis

    print('Ex1, delta north, east, down = {0}, {1}, {2}'.format(p_AB_N[0],
                                                                p_AB_N[1],
                                                                p_AB_N[2]))
    print('Ex1, azimuth = {0} deg'.format(deg(azimuth)))

    assert_allclose(p_AB_N[0], 331730.23478089)
    assert_allclose(p_AB_N[1], 332997.87498927)
    assert_allclose(p_AB_N[2], 17404.27136194)
    assert_allclose(deg(azimuth), 45.10926324)


def test_Ex2_B_and_delta_in_frame_B_to_C_in_frame_E():
    # delta vector from B to C, decomposed in B is given:
    p_BC_B = np.r_[3000, 2000, 100].reshape((-1, 1))  # pylint: disable=too-many-function-args

    # Position and orientation of B is given:
    n_EB_E = unit([[1], [2], [3]])  # unit to get unit length of vector
    z_EB = -400
    R_NB = zyx2R(rad(10), rad(20), rad(30))
    # the three angles are yaw, pitch, and roll

    # A custom reference ellipsoid is given (replacing WGS-84):
    a, f = 6378135, 1.0 / 298.26  # (WGS-72)

    # Find the position of C.
    # SOLUTION:
    # Step1: Find R_EN:
    R_EN = n_E2R_EN(n_EB_E)

    # Step2: Find R_EB, from R_EN and R_NB:
    R_EB = np.dot(R_EN, R_NB)  # Note: closest frames cancel

    # Step3: Decompose the delta vector in E:
    p_BC_E = np.dot(R_EB, p_BC_B)
    # no transpose of R_EB, since the vector is in B

    # Step4: Find the position of C, using the functions that goes from one
    # position and a delta, to a new position:
    n_EC_E, z_EC = n_EA_E_and_p_AB_E2n_EB_E(n_EB_E, p_BC_E, z_EB, a, f)

    # When displaying the resulting position for humans, it is more
    # convenient to see lat, long:
    lat_EC, long_EC = n_E2lat_lon(n_EC_E)
    # Here we also assume that the user wants output height (= - depth):
    msg = 'Ex2, Pos C: lat, long = {},{} deg,  height = {} m'
    print(msg.format(deg(lat_EC), deg(long_EC), -z_EC))

    assert_allclose(deg(lat_EC), 53.32637826)
    assert_allclose(deg(long_EC), 63.46812344)
    assert_allclose(z_EC, -406.00719607)


def test_Ex3_ECEF_vector_to_geodetic_latitude():

    # Position B is given as p_EB_E ("ECEF-vector")

    p_EB_E = 6371e3 * np.vstack((0.9, -1, 1.1))  # m

    # Find position B as geodetic latitude, longitude and height

    # SOLUTION:
    # Find n-vector from the p-vector:
    n_EB_E, z_EB = p_EB_E2n_EB_E(p_EB_E)

    # Convert to lat, long and height:
    lat_EB, long_EB = n_E2lat_lon(n_EB_E)
    h_EB = -z_EB
    msg = 'Ex3, Pos B: lat, long = {} {} deg, height = {} m'
    print(msg.format(deg(lat_EB), deg(long_EB), h_EB))

    assert_allclose(deg(lat_EB), 39.37874867)
    assert_allclose(deg(long_EB), -48.0127875)
    assert_allclose(h_EB, 4702059.83429485)


def test_Ex4_geodetic_latitude_to_ECEF_vector():

    # Position B is given with lat, long and height:
    lat_EB_deg = 1
    long_EB_deg = 2
    h_EB = 3

    # Find the vector p_EB_E ("ECEF-vector")

    # SOLUTION:
    # Step1: Convert to n-vector:
    n_EB_E = lat_lon2n_E(rad(lat_EB_deg), rad(long_EB_deg))

    # Step2: Find the ECEF-vector p_EB_E:
    p_EB_E = n_EB_E2p_EB_E(n_EB_E, -h_EB)

    print('Ex4: p_EB_E = {0} m'.format(p_EB_E.ravel()))

    assert_allclose(p_EB_E.ravel(), [6373290.27721828, 222560.20067474, 110568.82718179])


def test_Ex5_great_circle_distance():

    # Position A and B are given as n_EA_E and n_EB_E:
    # Enter elements as lat/long in deg:
    n_EA_E = lat_lon2n_E(rad(88), rad(0))
    n_EB_E = lat_lon2n_E(rad(89), rad(-170))

    r_Earth = 6371e3  # m, mean Earth radius

    # SOLUTION:
    s_AB = great_circle_distance(n_EA_E, n_EB_E, radius=r_Earth)
    d_AB = euclidean_distance(n_EA_E, n_EB_E, radius=r_Earth)

    msg = 'Ex5, Great circle distance = {} km, Euclidean distance = {} km'
    print(msg.format(s_AB / 1000, d_AB / 1000))

    assert_allclose(s_AB / 1000, 332.45644411)
    assert_allclose(d_AB / 1000, 332.41872486)


def test_Ex6_interpolated_position():

    # Position B at time t0 and t2 is given as n_EB_E_t0 and n_EB_E_t1:
    # Enter elements as lat/long in deg:
    n_EB_E_t0 = lat_lon2n_E(rad(89), rad(0))
    n_EB_E_t1 = lat_lon2n_E(rad(89), rad(180))

    # The times are given as:
    t0 = 10
    t1 = 20
    ti = 16  # time of interpolation

    # Find the interpolated position at time ti, n_EB_E_ti

    # SOLUTION:
    # Using standard interpolation:
    n_EB_E_ti = unit(n_EB_E_t0 + (ti - t0) * (n_EB_E_t1 - n_EB_E_t0) / (t1 - t0))

    # When displaying the resulting position for humans, it is more
    # convenient to see lat, long:
    lat_EB_ti, long_EB_ti = n_E2lat_lon(n_EB_E_ti)
    msg = 'Ex6, Interpolated position: lat, long = {} {} deg'
    print(msg.format(deg(lat_EB_ti), deg(long_EB_ti)))

    assert_allclose(deg(lat_EB_ti), 89.7999805)
    assert_allclose(deg(long_EB_ti), 180.)


def test_Ex7_mean_position():

    # Three positions A, B and C are given:
    # Enter elements as lat/long in deg:
    n_EA_E = lat_lon2n_E(rad(90), rad(0))
    n_EB_E = lat_lon2n_E(rad(60), rad(10))
    n_EC_E = lat_lon2n_E(rad(50), rad(-20))

    # Find the horizontal mean position:
    n_EM_E = unit(n_EA_E + n_EB_E + n_EC_E)

    truth = [0.3841171702926, -0.046602405485689447, 0.9221074857571395]
    # The result is best viewed with a figure that shows the n-vectors
    # relative to an Earth-model:
    # print('Ex7, See figure')
    # plot_earth_figure(n_EA_E,n_EB_E,n_EC_E,n_EM_E)
    # print(n_EM_E.ravel().tolist())
    assert_allclose(n_EM_E.ravel(), truth)
    # Alternatively:
    n_EM_E = mean_horizontal_position(np.hstack((n_EA_E, n_EB_E, n_EC_E)))
    # print(n_EM_E.ravel())
    assert_allclose(n_EM_E.ravel(), truth)


def test_Ex8_position_A_and_azimuth_and_distance_to_B():

    # Position A is given as n_EA_E:
    # Enter elements as lat/long in deg:
    lat, lon = rad(80), rad(-90)

    n_EA_E = lat_lon2n_E(lat, lon)

    # The initial azimuth and great circle distance (s_AB), and Earth
    # radius (r_Earth) are also given:
    azimuth = rad(200)
    s_AB = 1000  # m
    r_Earth = 6371e3  # m, mean Earth radius

    # Find the destination point B, as n_EB_E ("The direct/first geodetic
    # problem" for a sphere)

    # SOLUTION:
    # Step1: Convert distance in meter into distance in [rad]:
    distance_rad = s_AB / r_Earth
    # Step2: Find n_EB_E:
    n_EB_E = n_EA_E_distance_and_azimuth2n_EB_E(n_EA_E, distance_rad, azimuth)

    # When displaying the resulting position for humans, it is more
    # convenient to see lat, long:
    lat_EB, long_EB = n_E2lat_lon(n_EB_E)
    print('Ex8, Destination: lat, long = {0} {1} deg'.format(deg(lat_EB),
                                                             deg(long_EB)))

    assert_allclose(deg(lat_EB), 79.99154867)
    assert_allclose(deg(long_EB), -90.01769837)
    azimuth1 = n_EA_E_and_n_EB_E2azimuth(n_EA_E, n_EB_E, a=r_Earth, f=0)
    assert_allclose(azimuth, azimuth1 + 2 * np.pi)


def test_Ex9_intersect():

    # Two paths A and B are given by two pairs of positions:
    # Enter elements as lat/long in deg:
    n_EA1_E = lat_lon2n_E(rad(10), rad(20))
    n_EA2_E = lat_lon2n_E(rad(30), rad(40))
    n_EB1_E = lat_lon2n_E(rad(50), rad(60))
    n_EB2_E = lat_lon2n_E(rad(70), rad(80))

    # Find the intersection between the two paths, n_EC_E:
    n_EC_E_tmp = unit(np.cross(np.cross(n_EA1_E, n_EA2_E, axis=0),
                               np.cross(n_EB1_E, n_EB2_E, axis=0), axis=0))

    # n_EC_E_tmp is one of two solutions, the other is -n_EC_E_tmp. Select
    # the one that is closet to n_EA1_E, by selecting sign from the dot
    # product between n_EC_E_tmp and n_EA1_E:
    n_EC_E = np.sign(np.dot(n_EC_E_tmp.T, n_EA1_E)) * n_EC_E_tmp

    # When displaying the resulting position for humans, it is more
    # convenient to see lat, long:
    lat_EC, long_EC = n_E2lat_lon(n_EC_E)
    msg = 'Ex9, Intersection: lat, long = {} {} deg'
    print(msg.format(deg(lat_EC), deg(long_EC)))
    assert_allclose(deg(lat_EC), 40.31864307)
    assert_allclose(deg(long_EC), 55.90186788)


def test_Ex10_cross_track_distance():

    # Position A1 and A2 and B are given as n_EA1_E, n_EA2_E, and n_EB_E:
    # Enter elements as lat/long in deg:
    n_EA1_E = lat_lon2n_E(rad(0), rad(0))
    n_EA2_E = lat_lon2n_E(rad(10), rad(0))
    n_EB_E = lat_lon2n_E(rad(1), rad(0.1))

    radius = 6371e3  # m, mean Earth radius

    # Find the cross track distance from path A to position B.

    # SOLUTION:
    # Find the unit normal to the great circle:
    c_E = unit(np.cross(n_EA1_E, n_EA2_E, axis=0))
    # Find the great circle cross track distance:
    sin_theta = -np.dot(c_E.T, n_EB_E)  # pylint: disable=invalid-unary-operand-type
    s_xt = np.arcsin(sin_theta) * radius
    # ill conditioned for small angles:
    # s_xt2 = (np.arccos(-sin_theta) - np.pi / 2) * radius

    # Find the Euclidean cross track distance:
    d_xt = sin_theta * radius
    msg = 'Ex10, Cross track distance = {} m, Euclidean = {} m'
    print(msg.format(s_xt, d_xt))

    assert_allclose(s_xt, 11117.79911015)
    assert_allclose(d_xt, 11117.79346741)


def test_small_and_large_cross_track_distance():
    radius = 6371e3  # m, mean Earth radius
    n_EA1_E = lat_lon2n_E(rad(5), rad(10))
    n_EA2_E = lat_lon2n_E(rad(10), rad(10))
    n_EB0_E = lat_lon2n_E(rad(7), rad(10.1))

    path = (n_EA1_E, n_EA2_E)
    n_EB1_E = closest_point_on_great_circle(path, n_EB0_E)

    for s_xt0 in [np.pi / 3 * radius, 10., 0.1, 1e-4, 1e-8]:
        distance_rad = s_xt0 / radius
        n_EB_E = n_EA_E_distance_and_azimuth2n_EB_E(n_EB1_E, distance_rad, np.pi / 2)

        n_EB2_E = closest_point_on_great_circle(path, n_EB_E)
        s_xt = great_circle_distance(n_EB1_E, n_EB_E, radius)
        c_E = unit(np.cross(n_EA1_E, n_EA2_E, axis=0))
        s_xt2 = (np.arccos(np.dot(c_E.T, n_EB_E)) - np.pi / 2) * radius
        s_xt3 = cross_track_distance(path, n_EB_E, method='greatcircle', radius=radius)

        # pylint: disable=invalid-unary-operand-type
        s_xt4 = np.arctan2(-np.dot(c_E.T, n_EB_E),
                           np.linalg.norm(np.cross(c_E, n_EB_E, axis=0), axis=0)) * radius
        rtol = 10**(-min(9 + np.log10(s_xt0), 15))
        assert_allclose(n_EB2_E, n_EB1_E)
        assert_allclose(s_xt, s_xt0, rtol=rtol)
        assert_allclose(s_xt2, s_xt0, rtol=rtol)
        assert_allclose(s_xt3, s_xt0, rtol=rtol)
        assert_allclose(s_xt4, s_xt0, rtol=rtol)


def test_n_EA_E_and_n_EB_E2azimuth():
    n1_E = np.array([[0], [0], [1]])
    n2_E = np.array([[0], [1], [0]])
    n3_E = np.array([[0, 0], [0, 0], [1, 1]])
    n4_E = np.array([[0, 0], [1, 1], [0, 0]])
    for n1, n2 in zip((n1_E, n3_E, n3_E), (n2_E, n2_E, n4_E)):
        azimuth = n_EA_E_and_n_EB_E2azimuth(n1, n2)
        assert_allclose(azimuth, np.pi / 2)
        azimuth = n_EA_E_and_n_EB_E2azimuth(n2, n1)
        assert_allclose(azimuth, 0)
