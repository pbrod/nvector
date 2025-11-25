"""
Created on 18. des. 2015

@author: pab
"""

import sys
import warnings
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose as _assert_allclose  # @UnresolvedImport

import nvector as nv
from nvector.objects import FrameB, FrameE, FrameL, FrameN, GeoPath, GeoPoint, delta_L
from nvector.util import unit

assert_allclose = partial(_assert_allclose, atol=1e-15)

EARTH_RADIUS_M = 6371009.0


@pytest.mark.parametrize(
    "lat_a,lat_b,method",
    [
        (81, 80, "ellipsoid"),
        (79, 80, "ellipsoid"),
        (81, 80, "greatcircle"),
        (79, 80, "greatcircle"),
    ],
)
def test_geo_path_on_path(lat_a, lat_b, method):
    wgs84 = FrameE(name="WGS84")
    point_a = wgs84.GeoPointFromDegrees(latitude=lat_a, longitude=0)
    point_b = wgs84.GeoPointFromDegrees(latitude=lat_b, longitude=0)
    point_c = wgs84.GeoPointFromDegrees(latitude=0.5 * (lat_a + lat_b), longitude=0)

    path = nv.GeoPath(point_a, point_b)
    for point in [point_a, point_b, point_c]:
        assert path.on_path(point, method=method)

    point_a1 = wgs84.GeoPointFromDegrees(latitude=lat_a, longitude=1e-6)
    point_b1 = wgs84.GeoPointFromDegrees(latitude=lat_b, longitude=1e-6)
    for point in [point_a1, point_b1]:
        assert not path.on_path(point, method=method)

    tol = 1e-10
    lat_e, lat_f = (lat_a - tol, lat_b + tol) if lat_a < lat_b else (lat_a + tol, lat_b - tol)
    point_e = wgs84.GeoPointFromDegrees(latitude=lat_e, longitude=0)
    point_f = wgs84.GeoPointFromDegrees(latitude=lat_f, longitude=0)
    for point in [point_e, point_f]:
        assert not path.on_path(point, method=method)


def test_geo_path_repr():
    wgs84 = FrameE(name="WGS84")
    point_a = wgs84.GeoPointFromDegrees(latitude=81, longitude=0)
    point_b = wgs84.GeoPointFromDegrees(latitude=79, longitude=0)

    path = nv.GeoPath(point_a, point_b)
    r = repr(path)
    txt = (
        "GeoPath(point_a=GeoPoint(latitude=1.413716694115407, longitude=0.0, z=0, "
        "frame=FrameE(a=6378137.0, f=0.0033528106647474805, name='WGS84', axes='e')), "
        "point_b=GeoPoint(latitude=1.3788101090755203, longitude=0.0, z=0, "
        "frame=FrameE(a=6378137.0, f=0.0033528106647474805, name='WGS84', axes='e')))"
    )
    assert r == txt


class TestGeoPoint:
    def test_scalar_geopoint_to_nvector_to_geopoint(self):
        wgs84 = FrameE(name="WGS84")
        gp1 = wgs84.GeoPointFromDegrees(10, 5)
        np1 = gp1.to_nvector()
        gp2 = np1.to_geo_point()
        assert gp1 == gp2
        # Check that the round trip returns a scalar GeoPoint
        assert np.ndim(gp2.z) == 0
        assert np.ndim(gp2.latitude) == 0
        assert np.ndim(gp2.longitude) == 0

    def test_scalar_geopoint_to_nvector_to_ecef_to_geopoint(self):
        wgs84 = FrameE(name="WGS84")
        gp1 = wgs84.GeoPointFromDegrees(10, 5)
        np1 = gp1.to_nvector()
        ep1 = np1.to_ecef_vector()
        gp2 = ep1.to_geo_point()
        assert gp1 == gp2
        # Check that the round trip returns a scalar GeoPoint
        assert np.ndim(gp2.z) == 0
        assert np.ndim(gp2.latitude) == 0
        assert np.ndim(gp2.longitude) == 0

        assert np.ndim(ep1.length) == 0
        assert np.ndim(ep1.azimuth) == 0
        assert np.ndim(ep1.elevation) == 0

    def test_vector_geopoint_to_nvector_to_geopoint(self):
        wgs84 = FrameE(name="WGS84")
        gp1 = wgs84.GeoPointFromDegrees(10, [5, 7])
        np1 = gp1.to_nvector()
        gp2 = np1.to_geo_point()
        assert gp1 == gp2
        # Check that the round trip returns a vector GeoPoint
        assert np.ndim(gp2.z) == 1
        assert np.ndim(gp2.latitude) == 1
        assert np.ndim(gp2.longitude) == 1

    def test_vector_geopoint_to_nvector_to_ecef_to_geopoint(self):
        wgs84 = FrameE(name="WGS84")
        gp1 = wgs84.GeoPointFromDegrees([10, 15], 5)
        np1 = gp1.to_nvector()
        ep1 = np1.to_ecef_vector()
        gp2 = ep1.to_geo_point()
        assert gp1 == gp2
        # Check that the round trip returns a vector GeoPoint
        assert np.ndim(gp2.z) == 1
        assert np.ndim(gp2.latitude) == 1
        assert np.ndim(gp2.longitude) == 1

        assert np.ndim(ep1.length) == 1
        assert np.ndim(ep1.azimuth) == 1
        assert np.ndim(ep1.elevation) == 1

    def test_distance_and_azimuth(self):
        wgs84 = FrameE(name="WGS84")
        point1 = wgs84.GeoPointFromDegrees(latitude=-30, longitude=0)
        point2 = wgs84.GeoPointFromDegrees(latitude=29.9, longitude=179.8)
        s_12, _azi1, _azi2 = point1.distance_and_azimuth(point2)
        assert_allclose(s_12, 19989832.82761)

        n_a = point1.to_nvector()
        n_b = point2.to_nvector()
        s_ab, _azia, _azib = nv.geodesic_distance(n_a.normal, n_b.normal, wgs84.a, wgs84.f)
        assert_allclose(s_ab, 19989832.82761)

        point1 = wgs84.GeoPointFromDegrees(latitude=0, longitude=0)
        point2 = wgs84.GeoPointFromDegrees(latitude=0.5, longitude=179.5)
        s_12, _azi1, _azi2 = point1.distance_and_azimuth(point2)
        assert_allclose(s_12, 19936288.578965)

        n_a = point1.to_nvector()
        n_b = point2.to_nvector()
        s_ab, _azia, _azib = nv.geodesic_distance(n_a.normal, n_b.normal, wgs84.a, wgs84.f)
        # s_ab, azia, azib = nv.geodesic_distance(n_a.normal, n_b.normal, wgs84.a, wgs84.f)
        assert_allclose(s_ab, 19936288.578965)

        point1 = wgs84.GeoPointFromDegrees(latitude=88, longitude=0)
        point2 = wgs84.GeoPointFromDegrees(latitude=89, longitude=-170)
        s_12, azi1, azi2 = point1.distance_and_azimuth(point2)
        assert_allclose(s_12, 333947.509468)
        assert_allclose(nv.deg(azi1, azi2), (-3.3309161604062467, -173.327884597742))

        n_a = point1.to_nvector()
        n_b = point2.to_nvector()

        s_ab, _azia, _azib = nv.geodesic_distance(n_a.normal, n_b.normal, wgs84.a, wgs84.f)
        #         n_EA_E = nv.lat_lon2n_E(0,0)
        #         n_EB_E = nv.lat_lon2n_E(*nv.rad(0.5, 179.5))
        #         np.allclose(nv.geodesic_distance(n_EA_E, n_EB_E), 19909099.44101977)

        p3, azib = point1.displace(s_12, azi1)
        assert_allclose(nv.deg(azib), -173.327884597742)
        assert_allclose(p3.latlon_deg, (89, -170, 0))

        p4, azia = point2.displace(s_12, azi2 + np.pi)
        assert_allclose(nv.deg(azia), -3.3309161604062467 + 180)

        truth = (88, 0, 0)
        assert_allclose(p4.latlon_deg, truth, atol=1e-12)  # pylint: disable=redundant-keyword-arg

        # ------ greatcircle --------
        s_12, azi1, azi2 = point1.distance_and_azimuth(point2, method="greatcircle")
        assert_allclose(s_12, 331713.817039)
        assert_allclose(nv.deg(azi1, azi2), (-3.330916, -173.327885))

        p3, azib = point1.displace(s_12, azi1, method="greatcircle")
        assert_allclose(nv.deg(azib), -173.32784)
        assert_allclose(p3.latlon_deg, (89.000005, -169.999949, 0))

        p4, azia = point2.displace(s_12, azi2 + np.pi, method="greatcircle")
        _assert_allclose(p4.latlon_deg, truth, atol=1e-4)  # Less than 0.4 meters
        assert_allclose(nv.deg(azia), -3.3309161604062467 + 180)

    def test_displace(self):
        wgs84 = FrameE(name="WGS84")
        # Test example in Karney table 2:
        s_ab = 10000000
        azi_a = np.deg2rad(30)
        point1 = wgs84.GeoPointFromDegrees(latitude=40, longitude=0)
        point22, azi222 = point1.displace(s_ab, azi_a)
        nbb = point22.to_nvector()
        point2 = wgs84.GeoPointFromDegrees(latitude=41.79331020506, longitude=137.84490004377)
        s_12, azi1, azi2 = point1.distance_and_azimuth(point2)

        assert_allclose(s_12, s_ab)
        assert_allclose(azi1, azi_a)

        n_a = point1.to_nvector()
        n_b = point2.to_nvector()
        #         s_ab = nv.geodesic_distance(n_a.normal, n_b.normal, wgs84.a, wgs84.f)
        #         assert_allclose(s_12, 19909099.44101977)
        n_eb_e, azi22 = nv.geodesic_reckon(n_a.normal, s_ab, azi_a, wgs84.a, wgs84.f)
        assert_allclose(azi22, azi2)
        assert_allclose(azi222, azi2)
        # da = azi22 - azi2
        # dn = n_eb_e - n_b.normal
        # dn2 = n_eb_e - nbb.normal
        assert_allclose(n_eb_e, n_b.normal)
        assert_allclose(n_eb_e, nbb.normal)

    @pytest.mark.skipif(sys.platform == "darwin", reason="Fails on Mac for some reason!?")
    def test_geopoint_repr(self):
        wgs84 = FrameE(name="WGS84")
        point_a = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=3)
        r0 = repr(point_a)
        txt = (
            "GeoPoint(latitude=0.017453292519943295, longitude=0.03490658503988659, z=3, "
            "frame=FrameE(a=6378137.0, f=0.0033528106647474805, name='WGS84', axes='e'))"
        )
        assert r0 == txt

        point_b = wgs84.GeoPointFromDegrees(latitude=4, longitude=5, z=6)
        p_AB_N = point_a.delta_to(point_b)
        r = repr(p_AB_N)
        txt2 = (
            "Pvector(pvector=[[331730.2347808943], [332997.8749892696], [17404.271361936342]],"
            " frame=FrameN("
            "nvector=Nvector(normal=[[0.9992386149554826], [0.03489418134011367],"
            " [0.01745240643728351]], z=[0], "
            "frame=FrameE(a=6378137.0, f=0.0033528106647474805, "
            "name='WGS84', axes='e'))), scalar=True)"
        )
        assert r == txt2


class TestFrames:
    def test_compare_E_frames(self):
        E = FrameE(name="WGS84")
        E2 = FrameE(a=E.a, f=E.f)
        assert E == E2
        E3 = FrameE(a=E.a, f=0)
        assert E != E3

    def test_compare_B_frames(self):
        E = FrameE(name="WGS84")
        E2 = FrameE(name="WGS72")

        n_EB_E = E.Nvector(unit([[1], [2], [3]]), z=-400)
        B = FrameB(n_EB_E, yaw=10, pitch=20, roll=30, degrees=True)

        assert B != E

        B2 = FrameB(n_EB_E, yaw=1, pitch=20, roll=30, degrees=True)
        assert B != B2

        B3 = FrameB(n_EB_E, yaw=10, pitch=20, roll=30, degrees=True)
        assert B == B3

        n_EC_E = E.Nvector(unit([[1], [2], [2]]), z=-400)
        B4 = FrameB(n_EC_E, yaw=10, pitch=20, roll=30, degrees=True)
        assert B != B4

        n_ED_E = E2.Nvector(unit([[1], [2], [3]]), z=-400)
        B5 = FrameB(n_ED_E, yaw=10, pitch=20, roll=30, degrees=True)
        assert B != B5

    def test_compare_N_frames(self):
        wgs84 = FrameE(name="WGS84")
        wgs72 = FrameE(name="WGS72")
        pointA = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=3)
        pointB = wgs72.GeoPointFromDegrees(latitude=1, longitude=2, z=6)

        frame_N = FrameN.from_point(pointA)
        frame_L1 = FrameL.from_point(pointA, wander_azimuth=0)
        frame_L2 = FrameL.from_point(pointA, wander_azimuth=0)
        frame_L3 = FrameL.from_point(pointB, wander_azimuth=0)

        assert frame_N == frame_L1
        assert not (frame_N != frame_L1)

        assert frame_N == frame_L2

        assert frame_N != frame_L3
        assert frame_L1 != frame_L3

    def test_compare_L_frames(self):
        wgs84 = FrameE(name="WGS84")
        wgs72 = FrameE(name="WGS72")
        pointA = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=3)
        pointB = wgs72.GeoPointFromDegrees(latitude=1, longitude=2, z=6)

        frame_N = FrameL.from_point(pointA)
        frame_N1 = FrameL.from_point(pointA, wander_azimuth=10)
        frame_N2 = FrameL.from_point(pointB, wander_azimuth=10)

        assert frame_N != frame_N1
        assert frame_N != frame_N2
        assert frame_N1 != frame_N2

    @staticmethod
    def test_compute_delta_L_in_moving_frame_east():
        wgs84 = FrameE(name="WGS84")
        point_a = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=0)
        point_b = wgs84.GeoPointFromDegrees(latitude=1, longitude=2.005, z=0)
        sensor_position = wgs84.GeoPointFromDegrees(latitude=1.000090437, longitude=2.0025, z=0)
        path = GeoPath(point_a, point_b)
        ti = np.linspace(0, 1.0, 8)
        ship_positions0 = path.interpolate(ti[:-1])
        ship_positions1 = path.interpolate(ti[1:])
        headings = ship_positions0.delta_to(ship_positions1).azimuth_deg
        assert_allclose(headings, 90, rtol=1e-4)  # , decimal=4)

        ship_positions = path.interpolate(ti)

        delta = delta_L(ship_positions, sensor_position, wander_azimuth=np.pi / 2)

        x, y, z = delta.pvector
        azimuth = np.round(delta.azimuth_deg)
        # positive angle about down-axis

        true_x = [
            278.2566243359911,
            198.7547317612817,
            119.25283909376164,
            39.750946370747656,
            -39.75094637085409,
            -119.25283909387079,
            -198.75473176137066,
            -278.2566243360949,
        ]
        assert_allclose(x, true_x)  # decimal=3)
        assert_allclose(y, -10, rtol=1e-3)  # decimal=3)
        assert_allclose(azimuth, [-2.0, -3.0, -5.0, -14.0, -166.0, -175.0, -177.0, -178.0])
        _assert_allclose(z, 0, atol=1e-2)  # decimal=2)

    @staticmethod
    def test_compute_delta_N_in_moving_frame_east():
        wgs84 = FrameE(name="WGS84")
        point_a = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=0)
        point_b = wgs84.GeoPointFromDegrees(latitude=1, longitude=2.005, z=0)
        sensor_position = wgs84.GeoPointFromDegrees(latitude=1.0, longitude=2.0025, z=0)
        path = GeoPath(point_a, point_b)
        ti = np.linspace(0, 1.0, 8)
        ship_positions0 = path.interpolate(ti[:-1])
        ship_positions1 = path.interpolate(ti[1:])
        headings = ship_positions0.delta_to(ship_positions1).azimuth_deg
        assert_allclose(headings, 90, rtol=1e-4)  # , decimal=4)

        ship_positions = path.interpolate(ti)

        delta = ship_positions.delta_to(sensor_position)

        x, y, z = delta.pvector
        azimuth = np.round(delta.azimuth_deg)
        # positive angle about down-axis

        true_y = [
            278.2566243359911,
            198.7547317612817,
            119.25283909376164,
            39.750946370747656,
            -39.75094637085409,
            -119.25283909387079,
            -198.75473176137066,
            -278.2566243360949,
        ]
        _assert_allclose(x, 0, atol=1e-3)  # decimal=3)
        assert_allclose(y, true_y)
        _assert_allclose(z, 0, atol=1e-2)  # , decimal=2)
        n2 = len(azimuth) // 2
        assert_allclose(azimuth[:n2], 90)
        assert_allclose(azimuth[n2:], -90)

    @staticmethod
    def test_compute_delta_L_in_moving_frame_north():
        wgs84 = FrameE(name="WGS84")
        point_a = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=0)
        point_b = wgs84.GeoPointFromDegrees(latitude=1.005, longitude=2.0, z=0)
        sensor_position = wgs84.GeoPointFromDegrees(latitude=1.0025, longitude=2.0, z=0)
        path = GeoPath(point_a, point_b)
        ti = np.linspace(0, 1.0, 8)
        ship_positions0 = path.interpolate(ti[:-1])
        ship_positions1 = path.interpolate(ti[1:])
        headings = ship_positions0.delta_to(ship_positions1).azimuth_deg
        _assert_allclose(headings, 0, atol=1e-8)  # , decimal=8)

        ship_positions = path.interpolate(ti)

        delta0 = delta_L(ship_positions, sensor_position, wander_azimuth=0)
        delta = ship_positions.delta_to(sensor_position)
        assert_allclose(delta0.pvector, delta.pvector)

        x, y, z = delta.pvector
        azimuth = np.round(np.abs(delta.azimuth_deg))
        # positive angle about down-axis

        true_x = [
            276.436537069603,
            197.45466985931083,
            118.47280221160541,
            39.49093416312986,
            -39.490934249581684,
            -118.47280298990226,
            -197.454672021303,
            -276.4365413071498,
        ]
        assert_allclose(x, true_x)
        _assert_allclose(y, 0, atol=1e-8)  # , decimal=8)
        _assert_allclose(z, 0, atol=1e-2)  # , decimal=2)
        n2 = len(azimuth) // 2
        assert_allclose(azimuth[:n2], 0)
        assert_allclose(azimuth[n2:], 180)


class TestExamples:
    @staticmethod
    def test_Ex1_A_and_B_to_delta_in_frame_N():
        wgs84 = FrameE(name="WGS84")
        point_a = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=3)
        point_b = wgs84.GeoPointFromDegrees(latitude=4, longitude=5, z=6)

        # Find the exact vector between the two positions, given in meters
        # north, east, and down, i.e. find delta_N.

        # SOLUTION:
        delta = point_a.delta_to(point_b)
        x, y, z = delta.pvector
        azimuth = delta.azimuth_deg
        elevation = delta.elevation_deg

        assert_allclose(x, 331730.23478089)
        assert_allclose(y, 332997.87498927)
        assert_allclose(z, 17404.27136194)
        assert_allclose(azimuth, 45.10926324)
        assert_allclose(elevation, 2.12055861)

    @staticmethod
    def test_Ex2_B_and_delta_in_frame_B_to_C_in_frame_E():
        # delta vector from B to C, decomposed in B is given:

        # A custom reference ellipsoid is given (replacing WGS-84):
        wgs72 = FrameE(name="WGS72")

        # Position and orientation of B is given 400m above E:
        n_EB_E = wgs72.Nvector(unit([[1], [2], [3]]), z=-400)

        frame_B = FrameB(n_EB_E, yaw=10, pitch=20, roll=30, degrees=True)
        p_BC_B = frame_B.Pvector(np.r_[3000, 2000, 100].reshape((-1, 1)))

        p_BC_E = p_BC_B.to_ecef_vector()
        p_EB_E = n_EB_E.to_ecef_vector()
        p_EC_E = p_EB_E + p_BC_E

        pointC = p_EC_E.to_geo_point()

        lat, lon, z = pointC.latlon_deg
        # Here we also assume that the user wants output height (= - depth):

        assert_allclose(lat, 53.32637826)
        assert_allclose(lon, 63.46812344)
        assert_allclose(z, -406.00719607)

    @staticmethod
    def test_Ex3_ECEF_vector_to_geodetic_latitude():
        wgs84 = FrameE(name="WGS84")
        # Position B is given as p_EB_E ("ECEF-vector")
        position_B = 6371e3 * np.vstack((0.9, -1, 1.1))  # m
        p_EB_E = wgs84.ECEFvector(position_B)

        # Find position B as geodetic latitude, longitude and height
        pointB = p_EB_E.to_geo_point()
        lat, lon, h = pointB.latitude_deg, pointB.longitude_deg, -pointB.z

        assert_allclose(lat, 39.37874867)
        assert_allclose(lon, -48.0127875)
        assert_allclose(h, 4702059.83429485)

    @staticmethod
    def test_Ex4_geodetic_latitude_to_ECEF_vector():
        wgs84 = FrameE(name="WGS84")
        pointB = wgs84.GeoPointFromDegrees(latitude=1, longitude=2, z=-3)

        p_EB_E = pointB.to_ecef_vector()

        assert_allclose(
            p_EB_E.pvector.ravel(), [6373290.27721828, 222560.20067474, 110568.82718179]
        )

    @staticmethod
    def test_Ex5_great_circle_distance():
        frame_E = FrameE(a=6371e3, f=0)
        point_a = frame_E.GeoPointFromDegrees(latitude=88, longitude=0)
        point_b = frame_E.GeoPointFromDegrees(latitude=89, longitude=-170)
        s_AB, _azia, _azib = point_a.distance_and_azimuth(point_b)

        p_AB_E = point_b.to_ecef_vector() - point_a.to_ecef_vector()
        # The Euclidean distance is given by:
        d_AB = p_AB_E.length

        assert_allclose(s_AB / 1000, 332.45644411)
        assert_allclose(d_AB / 1000, 332.41872486)

    @staticmethod
    def test_alternative_great_circle_distance():
        frame_E = FrameE(a=6371e3, f=0)
        point_a = frame_E.GeoPointFromDegrees(latitude=88, longitude=0)
        point_b = frame_E.GeoPointFromDegrees(latitude=89, longitude=-170)
        path = GeoPath(point_a, point_b)

        s_AB = path.track_distance(method="greatcircle")
        d_AB = path.track_distance(method="euclidean")
        s1_AB = path.track_distance(method="exact")

        assert_allclose(s_AB / 1000, 332.45644411)
        assert_allclose(s1_AB / 1000, 332.45644411)
        assert_allclose(d_AB / 1000, 332.41872486)

    @staticmethod
    def test_exact_ellipsoidal_distance():
        wgs84 = FrameE(name="WGS84")
        pointA = wgs84.GeoPointFromDegrees(latitude=88, longitude=0)
        pointB = wgs84.GeoPointFromDegrees(latitude=89, longitude=-170)
        s_AB, _azia, _azib = pointA.distance_and_azimuth(pointB)

        p_AB_E = pointB.to_ecef_vector() - pointA.to_ecef_vector()
        # The Euclidean distance is given by:
        d_AB = p_AB_E.length

        assert_allclose(s_AB / 1000, 333.94750946834665)
        assert_allclose(d_AB / 1000, 333.90962112)

    @staticmethod
    def test_Ex6_interpolated_position():
        # Position B at time t0 and t2 is given as n_EB_E_t0 and n_EB_E_t1:
        # Enter elements as lat/long in deg:
        wgs84 = FrameE(name="WGS84")
        n_EB_E_t0 = wgs84.GeoPointFromDegrees(89, 0).to_nvector()
        n_EB_E_t1 = wgs84.GeoPointFromDegrees(89, 180).to_nvector()

        # The times are given as:
        t0 = 10.0
        t1 = 20.0
        ti = 16.0  # time of interpolation

        # Find the interpolated position at time ti, n_EB_E_ti

        # SOLUTION:
        # Using standard interpolation:
        ti_n = (ti - t0) / (t1 - t0)
        n_EB_E_ti = n_EB_E_t0 + ti_n * (n_EB_E_t1 - n_EB_E_t0)

        # When displaying the resulting position for humans, it is more
        # convenient to see lat, long:
        g_EB_E_ti = n_EB_E_ti.to_geo_point()
        lat_ti, lon_ti = g_EB_E_ti.latitude_deg, g_EB_E_ti.longitude_deg

        assert_allclose(lat_ti, 89.7999805)
        assert_allclose(lon_ti, 180.0)

        # Alternative solution
        path = GeoPath(n_EB_E_t0, n_EB_E_t1)

        g_EB_E_ti = path.interpolate(ti_n).to_geo_point()
        lat_ti, lon_ti = g_EB_E_ti.latitude_deg, g_EB_E_ti.longitude_deg

        assert_allclose(lat_ti, 89.7999805)
        assert_allclose(lon_ti, 180.0)

    @staticmethod
    def test_Ex7_mean_position():
        # Three positions A, B and C are given:
        # Enter elements directly:
        # n_EA_E=unit(np.vstack((1, 0, -2)))
        # n_EB_E=unit(np.vstack((-1, -2, 0)))
        # n_EC_E=unit(np.vstack((0, -2, 3)))

        # or input as lat/long in deg:
        points = GeoPoint.from_degrees(latitude=[90, 60, 50], longitude=[0, 10, -20])
        nvectors = points.to_nvector()
        nmean = nvectors.mean()
        n_EM_E = nmean.normal

        assert_allclose(
            n_EM_E.ravel(), [0.3841171702926, -0.046602405485689447, 0.9221074857571395]
        )

    @staticmethod
    def test_Ex8_position_A_and_azimuth_and_distance_to_B():
        frame = FrameE(a=EARTH_RADIUS_M, f=0)
        point_a = frame.GeoPointFromDegrees(latitude=80, longitude=-90)
        point_b, _azimuthb = point_a.displace(distance=1000, azimuth=200, degrees=True)
        pointB2, _azimuthb = point_a.displace(distance=1000, azimuth=np.deg2rad(200))
        assert_allclose(point_b.latlon, pointB2.latlon)

        lat_B, lon_B = point_b.latitude_deg, point_b.longitude_deg

        assert_allclose(lat_B, 79.99154867)
        assert_allclose(lon_B, -90.01769837)

    @staticmethod
    def test_Ex9_intersect():
        # Two paths A and B are given by two pairs of positions:
        point_a1 = GeoPoint.from_degrees(10, 20)
        point_a2 = GeoPoint.from_degrees(30, 40)
        point_b1 = GeoPoint.from_degrees(50, 60)
        point_b2 = GeoPoint.from_degrees(70, 80)
        path_a = GeoPath(point_a1, point_a2)
        path_b = GeoPath(point_b1, point_b2)

        point_c = path_a.intersect(path_b).to_geo_point()

        lat, lon = point_c.latitude_deg, point_c.longitude_deg

        assert_allclose(lat, 40.31864307)
        assert_allclose(lon, 55.90186788)

    def test_intersect_on_parallell_paths(self):
        # Two paths A and B are given by two pairs of positions:
        point_a1 = GeoPoint.from_degrees(10, 20)
        point_a2 = GeoPoint.from_degrees(30, 40)
        point_b1 = GeoPoint.from_degrees(10, 20)
        point_b2 = GeoPoint.from_degrees(30, 40)
        path_a = GeoPath(point_a1, point_a2)
        path_b = GeoPath(point_b1, point_b2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pointC = path_a.intersect(path_b).to_geo_point()

        lat, lon = pointC.latitude_deg, pointC.longitude_deg

        assert np.isnan(lat)
        assert np.isnan(lon)

    def test_Ex10_cross_track_distance(self):
        frame = FrameE(a=6371e3, f=0)
        # Position A1 and A2 and B as lat/long in deg:
        point_a1 = frame.GeoPointFromDegrees(0, 0)
        point_a2 = frame.GeoPointFromDegrees(10, 0)
        point_b = frame.GeoPointFromDegrees(1, 0.1)
        point_b2 = frame.GeoPointFromDegrees(11, 0.1)
        point_b3 = frame.GeoPointFromDegrees(-1, 0.1)

        path_a = GeoPath(point_a1, point_a2)

        # Find the cross track distance from path A to position B.
        s_xt = path_a.cross_track_distance(point_b, method="greatcircle")
        d_xt = path_a.cross_track_distance(point_b, method="euclidean")

        point_c = path_a.closest_point_on_great_circle(point_b)
        point_c2 = path_a.closest_point_on_great_circle(point_b2)
        point_c3 = path_a.closest_point_on_path(point_b2)
        point_c4 = path_a.closest_point_on_path(point_b3)
        s_xt2, _az_bc, _az_cb = point_b.distance_and_azimuth(point_c)
        assert_allclose(s_xt2, 11117.79911015)
        assert_allclose(s_xt, 11117.79911015)
        assert_allclose(d_xt, 11117.79346741)

        assert path_a.on_path(point_c)
        assert path_a.on_path(point_c, method="exact")

        assert not path_a.on_path(point_c2)
        assert not path_a.on_path(point_c2, method="exact")
        assert point_c3 == point_a2
        assert point_c4 == point_a1
