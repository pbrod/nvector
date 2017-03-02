"""
Created on 18. des. 2015

@author: pab
"""
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal  # @UnresolvedImport
from nvector import (FrameB, FrameE, FrameN, FrameL, GeoPoint, GeoPath, unit,
                     diff_positions)

EARTH_RADIUS_M = 6371009.0


class TestFrames(unittest.TestCase):
    def test_compare_E_frames(self):
        E = FrameE(name='WGS84')
        E2 = FrameE(a=E.a, f=E.f)
        self.assertEqual(E, E2)
        self.assertEqual(E, E)
        E3 = FrameE(a=E.a, f=0)
        self.assertNotEqual(E, E3)

    def test_compare_B_frames(self):
        E = FrameE(name='WGS84')
        E2 = FrameE(name='WGS72')

        n_EB_E = E.Nvector(unit([[1], [2], [3]]), z=-400)
        B = FrameB(n_EB_E, yaw=10, pitch=20, roll=30, degrees=True)
        self.assertEqual(B, B)
        self.assertNotEqual(B, E)

        B2 = FrameB(n_EB_E, yaw=1, pitch=20, roll=30, degrees=True)
        self.assertNotEqual(B, B2)

        B3 = FrameB(n_EB_E, yaw=10, pitch=20, roll=30, degrees=True)
        self.assertEqual(B, B3)

        n_EC_E = E.Nvector(unit([[1], [2], [2]]), z=-400)
        B4 = FrameB(n_EC_E, yaw=10, pitch=20, roll=30, degrees=True)
        self.assertNotEqual(B, B4)

# TODO: This test fails on python>2.7
#         n_ED_E = E2.Nvector(unit([[1], [2], [3]]), z=-400)
#         B5 = FrameB(n_ED_E, yaw=10, pitch=20, roll=30, degrees=True)
#         self.assertNotEqual(B, B5)

    def test_compare_N_frames(self):
        wgs84 = FrameE(name='WGS84')
        wgs72 = FrameE(name='WGS72')
        pointA = wgs84.GeoPoint(latitude=1, longitude=2, z=3, degrees=True)
        pointB = wgs72.GeoPoint(latitude=1, longitude=2, z=6, degrees=True)

        frame_N = FrameN(pointA)
        frame_L1 = FrameL(pointA, wander_azimuth=0)
        frame_L2 = FrameL(pointA, wander_azimuth=0)
        frame_L3 = FrameL(pointB, wander_azimuth=0)

        self.assertEqual(frame_N, frame_N)

        self.assertEqual(frame_N, frame_L1)

        # TODO: frame_N != frame_L1 returns True on python 2.7:
        # self.assertNotEqual(frame_N, frame_L1)

        self.assertEqual(frame_N, frame_L2)

        # TODO: Fails for python > 2.7
        # self.assertTrue(frame_N != frame_L3)
        # self.assertTrue(frame_L1 != frame_L3)

    def test_compare_L_frames(self):
        wgs84 = FrameE(name='WGS84')
        # wgs72 = FrameE(name='WGS72')
        pointA = wgs84.GeoPoint(latitude=1, longitude=2, z=3, degrees=True)
        # pointB = wgs72.GeoPoint(latitude=1, longitude=2, z=6, degrees=True)

        frame_N = FrameL(pointA)
        frame_N1 = FrameL(pointA, wander_azimuth=10)
        # frame_N2 = FrameL(pointB, wander_azimuth=10)

        self.assertEqual(frame_N, frame_N)
        self.assertNotEqual(frame_N, frame_N1)
        # self.assertNotEqual(frame_N, frame_N2)
        # self.assertNotEqual(frame_N1, frame_N2)


class TestExamples(unittest.TestCase):
    @staticmethod
    def test_Ex1_A_and_B_to_delta_in_frame_N():
        wgs84 = FrameE(name='WGS84')
        pointA = wgs84.GeoPoint(latitude=1, longitude=2, z=3, degrees=True)
        pointB = wgs84.GeoPoint(latitude=4, longitude=5, z=6, degrees=True)

        # Find the exact vector between the two positions, given in meters
        # north, east, and down, i.e. find p_AB_N.

        # SOLUTION:
        p_AB_E = diff_positions(pointA, pointB)  # (delta decomposed in E).

        frame_N = FrameN(pointA)
        p_AB_N = p_AB_E.change_frame(frame_N)
        p_AB_N = p_AB_N.pvector
        # Step5: Also find the direction (azimuth) to B, relative to north:
        azimuth = np.rad2deg(np.arctan2(p_AB_N[1], p_AB_N[0]))
        # positive angle about down-axis

        print('Ex1, delta north, east, down = {0}, {1}, {2}'.format(p_AB_N[0],
                                                                    p_AB_N[1],
                                                                    p_AB_N[2]))

        print('Ex1, azimuth = {0} deg'.format(azimuth))

        assert_array_almost_equal(p_AB_N[0], 331730.23478089)
        assert_array_almost_equal(p_AB_N[1], 332997.87498927)
        assert_array_almost_equal(p_AB_N[2], 17404.27136194)
        assert_array_almost_equal(azimuth, 45.10926324)

    @staticmethod
    def test_Ex2_B_and_delta_in_frame_B_to_C_in_frame_E():
        # delta vector from B to C, decomposed in B is given:

        # A custom reference ellipsoid is given (replacing WGS-84):
        wgs72 = FrameE(name='WGS72')

        # Position and orientation of B is given 400m above E:
        n_EB_E = wgs72.Nvector(unit([[1], [2], [3]]), z=-400)

        frame_B = FrameB(n_EB_E, yaw=10, pitch=20, roll=30, degrees=True)
        p_BC_B = frame_B.Pvector(np.r_[3000, 2000, 100].reshape((-1, 1)))

        p_BC_E = p_BC_B.to_ecef_vector()

        p_EB_E = n_EB_E.to_ecef_vector()
        p_EC_E = p_EB_E + p_BC_E
        pointC = p_EC_E.to_geo_point()

        lat_EC, long_EC = pointC.latitude_deg, pointC.longitude_deg
        z_EC = pointC.z
        # Here we also assume that the user wants output height (= - depth):
        msg = 'Ex2, Pos C: lat, long = {},{} deg,  height = {} m'
        print(msg.format(lat_EC, long_EC, -z_EC))

        assert_array_almost_equal(lat_EC, 53.32637826)
        assert_array_almost_equal(long_EC, 63.46812344)
        assert_array_almost_equal(z_EC, -406.00719607)

    @staticmethod
    def test_Ex3_ECEF_vector_to_geodetic_latitude():

        wgs84 = FrameE(name='WGS84')
        # Position B is given as p_EB_E ("ECEF-vector")
        position_B = 6371e3 * np.vstack((0.9, -1, 1.1))  # m
        p_EB_E = wgs84.ECEFvector(position_B)

        # Find position B as geodetic latitude, longitude and height
        pointB = p_EB_E.to_geo_point()
        lat, lon, h = pointB.latitude_deg, pointB.longitude_deg, -pointB.z

        msg = 'Ex3, Pos B: lat, lon = {} {} deg, height = {} m'
        print(msg.format(lat, lon, h))
        assert_array_almost_equal(lat, 39.37874867)
        assert_array_almost_equal(lon, -48.0127875)
        assert_array_almost_equal(h, 4702059.83429485)

    @staticmethod
    def test_Ex4_geodetic_latitude_to_ECEF_vector():
        wgs84 = FrameE(name='WGS84')
        pointB = wgs84.GeoPoint(latitude=1, longitude=2, z=-3, degrees=True)

        p_EB_E = pointB.to_ecef_vector()
        print('Ex4: p_EB_E = {0} m'.format(p_EB_E.pvector.ravel()))

        assert_array_almost_equal(p_EB_E.pvector.ravel(),
                                  [6373290.27721828, 222560.20067474,
                                   110568.82718179])

    @staticmethod
    def test_Ex5_great_circle_distance():
        frame_E = FrameE(a=6371e3, f=0)
        positionA = frame_E.GeoPoint(latitude=88, longitude=0, degrees=True)
        positionB = frame_E.GeoPoint(latitude=89, longitude=-170, degrees=True)
        s_AB, _azia, _azib = positionA.distance_and_azimuth(positionB)

        p_AB_E = positionB.to_ecef_vector() - positionA.to_ecef_vector()
        # The Euclidean distance is given by:
        d_AB = np.linalg.norm(p_AB_E.pvector, axis=0)

        msg = 'Ex5, Great circle distance = {} km, Euclidean distance = {} km'
        print(msg.format(s_AB / 1000, d_AB / 1000))

        assert_array_almost_equal(s_AB / 1000, 332.45644411)
        assert_array_almost_equal(d_AB / 1000, 332.41872486)

    @staticmethod
    def test_alternative_great_circle_distance():
        frame_E = FrameE(a=6371e3, f=0)
        positionA = frame_E.GeoPoint(latitude=88, longitude=0, degrees=True)
        positionB = frame_E.GeoPoint(latitude=89, longitude=-170, degrees=True)
        path = GeoPath(positionA, positionB)

        s_AB = path.track_distance(method='greatcircle')
        d_AB = path.track_distance(method='euclidean')

        msg = 'Ex5, Great circle distance = {} km, Euclidean distance = {} km'
        print(msg.format(s_AB / 1000, d_AB / 1000))

        assert_array_almost_equal(s_AB / 1000, 332.45644411)
        assert_array_almost_equal(d_AB / 1000, 332.41872486)

    @staticmethod
    def test_exact_ellipsoidal_distance():
        wgs84 = FrameE(name='WGS84')
        pointA = wgs84.GeoPoint(latitude=88, longitude=0, degrees=True)
        pointB = wgs84.GeoPoint(latitude=89, longitude=-170, degrees=True)
        s_AB, _azia, _azib = pointA.distance_and_azimuth(pointB)

        p_AB_E = pointB.to_ecef_vector() - pointA.to_ecef_vector()
        # The Euclidean distance is given by:
        d_AB = np.linalg.norm(p_AB_E.pvector, axis=0)

        msg = 'Ex5, Great circle distance = {} km, Euclidean distance = {} km'
        print(msg.format(s_AB / 1000, d_AB / 1000))

        assert_array_almost_equal(s_AB / 1000, 333.94750946834665)
        assert_array_almost_equal(d_AB / 1000, 333.90962112)

    @staticmethod
    def test_Ex6_interpolated_position():

        # Position B at time t0 and t2 is given as n_EB_E_t0 and n_EB_E_t1:
        # Enter elements as lat/long in deg:
        wgs84 = FrameE(name='WGS84')
        n_EB_E_t0 = wgs84.GeoPoint(89, 0, degrees=True).to_nvector()
        n_EB_E_t1 = wgs84.GeoPoint(89, 180, degrees=True).to_nvector()

        # The times are given as:
        t0 = 10.
        t1 = 20.
        ti = 16.  # time of interpolation

        # Find the interpolated position at time ti, n_EB_E_ti

        # SOLUTION:
        # Using standard interpolation:
        ti_n = (ti - t0) / (t1 - t0)
        n_EB_E_ti = n_EB_E_t0 + ti_n * (n_EB_E_t1 - n_EB_E_t0)

        # When displaying the resulting position for humans, it is more
        # convenient to see lat, long:
        g_EB_E_ti = n_EB_E_ti.to_geo_point()
        lat_ti, lon_ti = g_EB_E_ti.latitude_deg, g_EB_E_ti.longitude_deg
        msg = 'Ex6, Interpolated position: lat, long = {} deg, {} deg'
        print(msg.format(lat_ti, lon_ti))

        assert_array_almost_equal(lat_ti, 89.7999805)
        assert_array_almost_equal(lon_ti, 180.)

        # Alternative solution
        path = GeoPath(n_EB_E_t0, n_EB_E_t1)

        g_EB_E_ti = path.interpolate(ti_n).to_geo_point()
        lat_ti, lon_ti = g_EB_E_ti.latitude_deg, g_EB_E_ti.longitude_deg
        msg = 'Ex6, Interpolated position: lat, long = {} deg, {} deg'
        print(msg.format(lat_ti, lon_ti))

        assert_array_almost_equal(lat_ti, 89.7999805)
        assert_array_almost_equal(lon_ti, 180.)

    @staticmethod
    def test_Ex7_mean_position():

        # Three positions A, B and C are given:
        # Enter elements directly:
        # n_EA_E=unit(np.vstack((1, 0, -2)))
        # n_EB_E=unit(np.vstack((-1, -2, 0)))
        # n_EC_E=unit(np.vstack((0, -2, 3)))

        # or input as lat/long in deg:
        points = GeoPoint(latitude=[90, 60, 50], longitude=[0, 10, -20],
                          degrees=True)
        nvectors = points.to_nvector()
        nmean = nvectors.mean_horizontal_position()
        n_EM_E = nmean.normal
        assert_array_almost_equal(n_EM_E.ravel(),
                                  [0.384117, -0.046602, 0.922107])

    @staticmethod
    def test_Ex8_position_A_and_azimuth_and_distance_to_B():
        frame = FrameE(a=EARTH_RADIUS_M, f=0)
        pointA = frame.GeoPoint(latitude=80, longitude=-90, degrees=True)
        pointB, _azimuthb = pointA.geo_point(distance=1000, azimuth=200,
                                             degrees=True)

        lat_B, lon_B = pointB.latitude_deg, pointB.longitude_deg

        print('Ex8, Destination: lat, long = {0} {1} deg'.format(lat_B, lon_B))
        assert_array_almost_equal(lat_B, 79.99154867)
        assert_array_almost_equal(lon_B, -90.01769837)

    @staticmethod
    def test_Ex9_intersect():

        # Two paths A and B are given by two pairs of positions:
        pointA1 = GeoPoint(10, 20, degrees=True)
        pointA2 = GeoPoint(30, 40, degrees=True)
        pointB1 = GeoPoint(50, 60, degrees=True)
        pointB2 = GeoPoint(70, 80, degrees=True)
        pathA = GeoPath(pointA1, pointA2)
        pathB = GeoPath(pointB1, pointB2)

        pointC = pathA.intersect(pathB).to_geo_point()

        lat, lon = pointC.latitude_deg, pointC.longitude_deg
        msg = 'Ex9, Intersection: lat, long = {} {} deg'
        print(msg.format(lat, lon))
        assert_array_almost_equal(lat, 40.31864307)
        assert_array_almost_equal(lon, 55.90186788)

    def test_intersection_of_parallell_paths(self):

        # Two paths A and B are given by two pairs of positions:
        pointA1 = GeoPoint(10, 20, degrees=True)
        pointA2 = GeoPoint(30, 40, degrees=True)
        pointB1 = GeoPoint(10, 20, degrees=True)
        pointB2 = GeoPoint(30, 40, degrees=True)
        pathA = GeoPath(pointA1, pointA2)
        pathB = GeoPath(pointB1, pointB2)

        pointC = pathA.intersect(pathB).to_geo_point()

        lat, lon = pointC.latitude_deg, pointC.longitude_deg
        msg = 'Ex9, Intersection: lat, long = {} {} deg'
        print(msg.format(lat, lon))
        self.assertTrue(np.isnan(lat))
        self.assertTrue(np.isnan(lon))

    @staticmethod
    def test_Ex10_cross_track_distance():

        frame = FrameE(a=6371e3, f=0)
        # Position A1 and A2 and B as lat/long in deg:
        pointA1 = frame.GeoPoint(0, 0, degrees=True)
        pointA2 = frame.GeoPoint(10, 0, degrees=True)
        pointB = frame.GeoPoint(1, 0.1, degrees=True)

        pathA = GeoPath(pointA1, pointA2)

        # Find the cross track distance from path A to position B.
        s_xt = pathA.cross_track_distance(pointB, method='greatcircle')
        d_xt = pathA.cross_track_distance(pointB, method='euclidean')
        msg = 'Ex10, Cross track distance = {} m, Euclidean = {} m'
        print(msg.format(s_xt, d_xt))

        assert_array_almost_equal(s_xt, 11117.79911015)
        assert_array_almost_equal(d_xt, 11117.79346741)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
