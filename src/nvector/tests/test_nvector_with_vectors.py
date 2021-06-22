'''
Created on 10. jun. 2021

@author: pab
'''
from functools import partial
import numpy as np
from nvector.util import unit, deg, rad
from nvector.core import (lat_lon2n_E, n_E2lat_lon,
                          n_EA_E_and_n_EB_E2p_AB_E,
                          n_EA_E_and_p_AB_E2n_EB_E,
                          p_EB_E2n_EB_E,
                          n_EB_E2p_EB_E,
                          great_circle_distance,
                          great_circle_distance_rad,
                          euclidean_distance,
                          cross_track_distance,
                          closest_point_on_great_circle,
                          course_over_ground,
                          on_great_circle_path,
                          n_EA_E_distance_and_azimuth2n_EB_E,
                          n_EA_E_and_n_EB_E2azimuth)
from numpy.testing import assert_allclose as _assert_allclose  # @UnresolvedImport

assert_allclose = partial(_assert_allclose, atol=1e-8)


def test_lat_lon2n_E():
    n_E = lat_lon2n_E(rad([0, 90]), [0, 0])
    print(n_E.tolist())
    assert_allclose(n_E, [[1.0, 0.0],
                          [0.0, 0.0],
                          [0.0, 1.0]])


def test_n_E2lat_lon():
    lat, lon = n_E2lat_lon([[1.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 1.0]])
    assert_allclose(lat, [0, np.pi / 2])
    assert_allclose(lon % np.pi, [0, 0])


def test_n_EA_E_and_n_EB_E2p_AB_E():
    a = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]
    b = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]
    c = [[1.0],
         [0.0],
         [0.0]]
    delta = n_EA_E_and_n_EB_E2p_AB_E(a, b)
    print(delta.tolist())
    assert_allclose(delta, [[0, 0],
                            [0, 0],
                            [0, 0]])
    # test broadcasting
    delta = n_EA_E_and_n_EB_E2p_AB_E(a, c)
    print(delta.tolist())
    assert_allclose(delta, [[0.0, 6378137.000000001],
                            [0.0, 0.0],
                            [0.0, -6356752.314245179]])


def test_n_EA_E_and_p_AB_E2n_EB_E():
    a = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]

    c = [[1.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    aa, depth = n_EA_E_and_p_AB_E2n_EB_E(a, [[0],
                                             [0],
                                             [0]])
    assert_allclose(aa, a)
    assert_allclose([0, 0], depth)
    cc, depth = n_EA_E_and_p_AB_E2n_EB_E(a, [[0.0, 6378137.000000001],
                                             [0.0, 0.0],
                                             [0.0, -6356752.314245179]])
    assert_allclose(cc, c)
    assert_allclose([0, 0], depth)


def test_p_EB_E2n_EB_E():

    b = [[6378137.000000001, 0.0],
         [0.0, 0.0],
         [0.0, 6356752.314245179]]
    a, depth = p_EB_E2n_EB_E(b)
    print(a.tolist())
    assert_allclose(a, [[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
    assert_allclose([0, 0], depth)


def test_n_EB_E2p_EB_E():
    a = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]
    b = n_EB_E2p_EB_E(a, depth=0)
    print(b.tolist())
    assert_allclose(b, [[6378137.000000001, 0.0],
                        [0.0, 0.0],
                        [0.0, 6356752.314245179]])


def test_great_circle_distance():
    a = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]

    c = [[1.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    distance = great_circle_distance(a, c)
    distance2 = great_circle_distance(c, a)
    # print(distance.tolist())
    assert_allclose(distance, [0.0, 10007557.535177227])
    assert_allclose(distance2, [0.0, 10007557.535177227])


def test_euclidean_distance():
    a = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]

    c = [[1.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    distance = euclidean_distance(a, c)
    print(distance.tolist())
    assert_allclose(distance, [0.0, 9009967.33380105])


def test_cross_track_distance():
    a = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]

    b = [[0.0, 0.0],
         [1.0, 1.0],
         [0.0, 0.0]]
    c = unit([[0.0, 1.0],
              [1.0, 1.0],
              [1.0, 0.0]])
    path = (a, b)
    radius = 6371e3  # mean earth radius [m]
    distance = cross_track_distance(path, c, radius=radius)
    # print(distance.tolist())
    assert_allclose(distance, [-5003771.699005142, 5003771.699005142])


def test_closest_point_on_great_circle():
    a = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]

    b = [[0.0, 0.0],
         [1.0, 1.0],
         [0.0, 0.0]]
    c = unit([[0.0, 1.0],
              [1.0, 1.0],
              [1.0, 0.0]])
    path = (a, b)
    radius = 6371e3  # mean earth radius [m]
    d = closest_point_on_great_circle(path, c)
    print(d.tolist())
    assert_allclose(d, [[0.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 0.0]])
    on = on_great_circle_path(path, d, radius)
    assert on.tolist() == [True, True]


def test_n_EA_E_distance_and_azimuth2n_EB_E():
    a = [[1.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0]]  # lat = [0, 0] and lon = [0, 0] degrees
    distance_rad = np.pi/2
    azimuth = np.r_[np.pi/2, 0]
    b = n_EA_E_distance_and_azimuth2n_EB_E(a, distance_rad, azimuth)
    print(deg(*n_E2lat_lon(a)))
    print(deg(*n_E2lat_lon(b)))
    print(b.tolist())
    assert_allclose(b, [[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])  # # lat = [0, 90] and lon = [90, 0] degrees

    aa = n_EA_E_distance_and_azimuth2n_EB_E(b, distance_rad, azimuth + np.pi)
    print(aa.tolist())
    assert_allclose(aa, a)


def test_n_EA_E_and_n_EB_E2azimuth():
    a = [[1.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0]]

    b = [[0.0, 0.0],
         [1.0, 0.0],
         [0.0, 1.0]]

    azimuth = n_EA_E_and_n_EB_E2azimuth(a, b)
    azimuth2 = n_EA_E_and_n_EB_E2azimuth(b, a)
    distance_rad = great_circle_distance_rad(a, b)

    print(azimuth, azimuth2)
    print(distance_rad)
    assert_allclose(azimuth, [np.pi/2, 0])
    assert_allclose(azimuth2, [-np.pi/2, np.pi])
    assert_allclose(distance_rad, [np.pi/2, np.pi/2])


def test_course_over_ground():
    a = [[1.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    b = [[0.0, 1.0],
         [1.0, 0.0],
         [0.0, 0.0]]
    c = [[1],
         [0],
         [0]]
    cog_a = course_over_ground(a)
    cog_b = course_over_ground(b)
    cog_c = course_over_ground(c)

    assert_allclose(cog_a, [0, 0])
    assert_allclose(cog_b, [-np.pi/2, -np.pi/2])
    assert np.isnan(cog_c)


if __name__ == '__main__':
    pass
