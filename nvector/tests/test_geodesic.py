import unittest

import numpy as np
from nvector import GeoPoint, FrameE

wgs84 = FrameE(name='WGS84')


TESTCASES = [
    [35.60777, -139.44815, 111.098748429560326,
     -11.17491, -69.95921, 129.289270889708762,
     8935244.5604818305, 80.50729714281974, 6273170.2055303837,
     0.16606318447386067, 0.16479116945612937, 12841384694976.432],
    [55.52454, 106.05087, 22.020059880982801,
     77.03196, 197.18234, 109.112041110671519,
     4105086.1713924406, 36.892740690445894, 3828869.3344387607,
     0.80076349608092607, 0.80101006984201008, 61674961290615.615],
    [-21.97856, 142.59065, -32.44456876433189,
     41.84138, 98.56635, -41.84359951440466,
     8394328.894657671, 75.62930491011522, 6161154.5773110616,
     0.24816339233950381, 0.24930251203627892, -6637997720646.717],
    [-66.99028, 112.2363, 173.73491240878403,
     -12.70631, 285.90344, 2.512956620913668,
     11150344.2312080241, 100.278634181155759, 6289939.5670446687,
     -0.17199490274700385, -0.17722569526345708, -121287239862139.744],
    [-17.42761, 173.34268, -159.033557661192928,
     -15.84784, 5.93557, -20.787484651536988,
     16076603.1631180673, 144.640108810286253, 3732902.1583877189,
     -0.81273638700070476, -0.81299800519154474, 97825992354058.708],
    [32.84994, 48.28919, 150.492927788121982,
     -56.28556, 202.29132, 48.113449399816759,
     16727068.9438164461, 150.565799985466607, 3147838.1910180939,
     -0.87334918086923126, -0.86505036767110637, -72445258525585.010],
    [6.96833, 52.74123, 92.581585386317712,
     -7.39675, 206.17291, 90.721692165923907,
     17102477.2496958388, 154.147366239113561, 2772035.6169917581,
     -0.89991282520302447, -0.89986892177110739, -1311796973197.995],
    [-50.56724, -16.30485, -105.439679907590164,
     -33.56571, -94.97412, -47.348547835650331,
     6455670.5118668696, 58.083719495371259, 5409150.7979815838,
     0.53053508035997263, 0.52988722644436602, 41071447902810.047],
    [-58.93002, -8.90775, 140.965397902500679,
     -8.91104, 133.13503, 19.255429433416599,
     11756066.0219864627, 105.755691241406877, 6151101.2270708536,
     -0.26548622269867183, -0.27068483874510741, -86143460552774.735],
    [-68.82867, -74.28391, 93.774347763114881,
     -50.63005, -8.36685, 34.65564085411343,
     3956936.926063544, 35.572254987389284, 3708890.9544062657,
     0.81443963736383502, 0.81420859815358342, -41845309450093.787],
    [-10.62672, -32.0898, -86.426713286747751,
     5.883, -134.31681, -80.473780971034875,
     11470869.3864563009, 103.387395634504061, 6184411.6622659713,
     -0.23138683500430237, -0.23155097622286792, 4198803992123.548],
    [-21.76221, 166.90563, 29.319421206936428,
     48.72884, 213.97627, 43.508671946410168,
     9098627.3986554915, 81.963476716121964, 6299240.9166992283,
     0.13965943368590333, 0.14152969707656796, 10024709850277.476],
    [-19.79938, -174.47484, 71.167275780171533,
     -11.99349, -154.35109, 65.589099775199228,
     2319004.8601169389, 20.896611684802389, 2267960.8703918325,
     0.93427001867125849, 0.93424887135032789, -3935477535005.785],
    [-11.95887, -116.94513, 92.712619830452549,
     4.57352, 7.16501, 78.64960934409585,
     13834722.5801401374, 124.688684161089762, 5228093.177931598,
     -0.56879356755666463, -0.56918731952397221, -9919582785894.853],
    [-87.85331, 85.66836, -65.120313040242748,
     66.48646, 16.09921, -4.888658719272296,
     17286615.3147144645, 155.58592449699137, 2635887.4729110181,
     -0.90697975771398578, -0.91095608883042767, 42667211366919.534],
    [1.74708, 128.32011, -101.584843631173858,
     -11.16617, 11.87109, -86.325793296437476,
     12942901.1241347408, 116.650512484301857, 5682744.8413270572,
     -0.44857868222697644, -0.44824490340007729, 10763055294345.653],
    [-25.72959, -144.90758, -153.647468693117198,
     -57.70581, -269.17879, -48.343983158876487,
     9413446.7452453107, 84.664533838404295, 6356176.6898881281,
     0.09492245755254703, 0.09737058264766572, 74515122850712.444],
    [-41.22777, 122.32875, 14.285113402275739,
     -7.57291, 130.37946, 10.805303085187369,
     3812686.035106021, 34.34330804743883, 3588703.8812128856,
     0.82605222593217889, 0.82572158200920196, -2456961531057.857],
    [11.01307, 138.25278, 79.43682622782374,
     6.62726, 247.05981, 103.708090215522657,
     11911190.819018408, 107.341669954114577, 6070904.722786735,
     -0.29767608923657404, -0.29785143390252321, 17121631423099.696],
    [-29.47124, 95.14681, -163.779130441688382,
     -27.46601, -69.15955, -15.909335945554969,
     13487015.8381145492, 121.294026715742277, 5481428.9945736388,
     -0.51527225545373252, -0.51556587964721788, 104679964020340.318]]


class GeodesicTest(unittest.TestCase):

    def test_inverse(self):
        options = dict(frame=wgs84, degrees=True)
        for l in TESTCASES:
            (lat1, lon1, azi1, lat2, lon2, azi2, s12) = l[:7]
            point1 = GeoPoint(lat1, lon1, **options)
            point2 = GeoPoint(lat2, lon2, **options)
            s_ab, az_a, az_b = point1.distance_and_azimuth(point2,
                                                           long_unroll=True,
                                                           degrees=True)

            self.assertAlmostEqual(azi1, az_a, delta=1e-13)
            self.assertAlmostEqual(azi2, az_b, delta=1e-13)
            self.assertAlmostEqual(s12, s_ab, delta=1e-8)

    def test_direct(self):
        options = dict(frame=wgs84, degrees=True)
        for l in TESTCASES:
            (lat1, lon1, azi1, lat2, lon2, azi2, s12) = l[:7]
            point1 = GeoPoint(lat1, lon1, **options)
            point2, az_b = point1.geo_point(s12, azi1, long_unroll=True,
                                            degrees=True)

            lat_b, lon_b = point2.latitude_deg, point2.longitude_deg
            self.assertAlmostEqual(lat2, lat_b, delta=1e-13)
            self.assertAlmostEqual(lon2, lon_b, delta=1e-13)
            self.assertAlmostEqual(azi2, az_b, delta=1e-13)


class GeodSolveTest(unittest.TestCase):

    def test_GeodSolve0(self):
        point1 = GeoPoint(40.6, -73.8, frame=wgs84, degrees=True)
        point2 = GeoPoint(49.01666667, 2.55, frame=wgs84, degrees=True)
        s_ab, az_a, az_b = point1.distance_and_azimuth(point2, degrees=True)
        self.assertAlmostEqual(az_a, 53.47022, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 111.59367, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 5853226, delta=0.5)

    def test_GeodSolve1(self):
        lat1, lon1, az1 = np.deg2rad((40.63972222, -73.77888889, 53.5,))
        lat_b, lon_b, az_b = wgs84.direct(lat1, lon1, az1, 5850e3)
        lat_b, lon_b, az_b = np.rad2deg((lat_b, lon_b, az_b))
        self.assertAlmostEqual(lat_b, 49.01467, delta=0.5e-5)
        self.assertAlmostEqual(lon_b, 2.56106, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 111.62947, delta=0.5e-5)

    def test_GeodSolve2(self):
        # Check fix for antipodal prolate bug found 2010-09-04
        geod = FrameE(6.4e6, -1 / 150.0)
        lat1, lon1, lat2, lon2 = np.deg2rad((0.07476, 0, -0.07476, 180))
        s_ab, az_a, az_b = geod.inverse(lat1, lon1, lat2, lon2)
        az_a, az_b = np.rad2deg((az_a, az_b))
        self.assertAlmostEqual(az_a, 90.00078, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 90.00078, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 20106193, delta=0.5)
        lat1, lon1, lat2, lon2 = np.deg2rad((0.1, 0, -0.1, 180))
        s_ab, az_a, az_b = geod.inverse(lat1, lon1, lat2, lon2)
        az_a, az_b = np.rad2deg((az_a, az_b))
        self.assertAlmostEqual(az_a, 90.00105, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 90.00105, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 20106193, delta=0.5)

    def test_GeodSolve4(self):
        # Check fix for short line bug found 2010-05-21
        lat1, lon1, lat2, lon2 = np.deg2rad((36.493349428792, 0,
                                             36.49334942879201, .0000008))
        s_ab, _az_a, _az_b = wgs84.inverse(lat1, lon1, lat2, lon2)

        self.assertAlmostEqual(s_ab, 0.072, delta=0.5e-3)

    def test_GeodSolve5(self):
        # Check fix for point2=pole bug found 2010-05-03
        lat1, lon1, az1 = np.deg2rad((0.01777745589997, 30, 0))
        lat_b, lon_b, az_b = wgs84.direct(lat1, lon1, az1, 10e6)
        lat_b, lon_b, az_b = np.rad2deg((lat_b, lon_b, az_b))
        self.assertAlmostEqual(lat_b, 90, delta=0.5e-5)
        if lon_b < 0:
            self.assertAlmostEqual(lon_b, -150, delta=0.5e-5)
            self.assertAlmostEqual(az_b, -180, delta=0.5e-5)
        else:
            self.assertAlmostEqual(lon_b, 30, delta=0.5e-5)
            self.assertAlmostEqual(az_b, 0, delta=0.5e-5)

    def test_GeodSolve6(self):
        # Check fix for volatile sbet12a bug found 2011-06-25 (gcc 4.4.4
        # x86 -O3).  Found again on 2012-03-27 with tdm-mingw32 (g++ 4.6.1).

        lat1, lon1, lat2, lon2 = np.deg2rad((88.202499451857, 0,
                                             -88.202499451857,
                                             179.981022032992859592))
        s_ab, _az_a, _az_b = wgs84.inverse(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(s_ab, 20003898.214, delta=0.5e-3)

        lat1, lon1, lat2, lon2 = np.deg2rad((89.262080389218, 0,
                                             -89.262080389218,
                                             179.992207982775375662))
        s_ab, _az_a, _az_b = wgs84.inverse(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(s_ab, 20003925.854, delta=0.5e-3)

        lat1, lon1, lat2, lon2 = np.deg2rad((89.333123580033, 0,
                                             -89.333123580032997687,
                                             179.99295812360148422))
        s_ab, _az_a, _az_b = wgs84.inverse(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(s_ab, 20003926.881, delta=0.5e-3)

    def test_GeodSolve9(self):
        # Check fix for volatile x bug found 2011-06-25 (gcc 4.4.4 x86 -O3)
        lat1, lon1, lat2, lon2 = np.deg2rad((56.320923501171, 0,
                                             -56.320923501171,
                                             179.664747671772880215))
        s_ab, _az_a, _az_b = wgs84.inverse(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(s_ab, 19993558.287, delta=0.5e-3)

    def test_GeodSolve10(self):
        # Check fix for adjust tol1_ bug found 2011-06-25 (Visual Studio
        # 10 rel + debug)
        lat1, lon1, lat2, lon2 = np.deg2rad((52.784459512564, 0,
                                             -52.784459512563990912,
                                             179.634407464943777557))
        s_ab, _az_a, _az_b = wgs84.inverse(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(s_ab, 19991596.095, delta=0.5e-3)

    def test_GeodSolve11(self):
        # Check fix for bet2 = -bet1 bug found 2011-06-25 (Visual Studio
        # 10 rel + debug)
        lat1, lon1, lat2, lon2 = np.deg2rad((48.522876735459, 0,
                                             -48.52287673545898293,
                                             179.599720456223079643))
        s_ab, _az_a, _az_b = wgs84.inverse(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(s_ab, 19989144.774, delta=0.5e-3)

    def test_GeodSolve12(self):
        # Check fix for inverse geodesics on extreme prolate/oblate
        # ellipsoids Reported 2012-08-29 Stefan Guenther
        # <stefan.gunther@embl.de>; fixed 2012-10-07
        geod = FrameE(89.8, -1.83)
        lat1, lon1, lat2, lon2 = np.deg2rad((0, 0, -10, 160))
        s_ab, az_a, az_b = geod.inverse(lat1, lon1, lat2, lon2)
        az_a, az_b = np.rad2deg((az_a, az_b))
        self.assertAlmostEqual(az_a, 120.27, delta=1e-2)
        self.assertAlmostEqual(az_b, 105.15, delta=1e-2)
        self.assertAlmostEqual(s_ab, 266.7, delta=1e-1)

    def test_GeodSolve14(self):
        # Check fix for inverse ignoring lon12 = nan
        s_ab, az_a, az_b = wgs84.inverse(0, 0, 1, np.nan, degrees=True)
        self.assertTrue(np.isnan(az_a))
        self.assertTrue(np.isnan(az_b))
        self.assertTrue(np.isnan(s_ab))

    def test_GeodSolve17(self):
        # Check fix for LONG_UNROLL bug found on 2015-05-07
        lat_b, lon_b, az_b = wgs84.direct(40, -75, -10, 2e7, long_unroll=True,
                                          degrees=True)

        self.assertAlmostEqual(lat_b, -39, delta=1)
        self.assertAlmostEqual(lon_b, -254, delta=1)
        self.assertAlmostEqual(az_b, -170, delta=1)

        lat_b, lon_b, az_b = wgs84.direct(40, -75, -10, 2e7, degrees=True)
        self.assertAlmostEqual(lat_b, -39, delta=1)
        self.assertAlmostEqual(lon_b, 105, delta=1)
        self.assertAlmostEqual(az_b, -170, delta=1)

    def test_GeodSolve29(self):
        # Check longitude unrolling with inverse calculation 2015-09-16
        s_ab, _az_a, _az_b = wgs84.inverse(0, 539, 0, 181, degrees=True)

        self.assertAlmostEqual(s_ab, 222639, delta=0.5)
        s_ab, _az_a, _az_b = wgs84.inverse(0, 539, 0, 181, degrees=True)
        self.assertAlmostEqual(s_ab, 222639, delta=0.5)

    def test_GeodSolve33(self):
        # Check max(-0.0,+0.0) issues 2015-08-22 (triggered by bugs in
        # Octave -- sind(-0.0) = +0.0 -- and in some version of Visual
        # Studio -- fmod(-0.0, 360.0) = +0.0.
        s_ab, az_a, az_b = wgs84.inverse(0, 0, 0, 179, degrees=True)
        self.assertAlmostEqual(az_a, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 19926189, delta=0.5)

        s_ab, az_a, az_b = wgs84.inverse(0, 0, 0, 179.5, degrees=True)
        self.assertAlmostEqual(az_a, 55.96650, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 124.03350, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 19980862, delta=0.5)

        s_ab, az_a, az_b = wgs84.inverse(0, 0, 0, 180, degrees=True)
        self.assertAlmostEqual(az_a, 0.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, -180.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 20003931, delta=0.5)

        s_ab, az_a, az_b = wgs84.inverse(0, 0, 1, 180, degrees=True)
        self.assertAlmostEqual(az_a, 0.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, -180.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 19893357, delta=0.5)

        geod = FrameE(6.4e6, 0)
        s_ab, az_a, az_b = geod.inverse(0, 0, 0, 179, degrees=True)
        self.assertAlmostEqual(az_a, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 19994492, delta=0.5)
        s_ab, az_a, az_b = geod.inverse(0, 0, 0, 180, degrees=True)
        self.assertAlmostEqual(az_a, 0.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, -180.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 20106193, delta=0.5)
        s_ab, az_a, az_b = geod.inverse(0, 0, 1, 180, degrees=True)
        self.assertAlmostEqual(az_a, 0.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, -180.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 19994492, delta=0.5)
        geod = FrameE(6.4e6, -1 / 300.0)
        s_ab, az_a, az_b = geod.inverse(0, 0, 0, 179, degrees=True)
        self.assertAlmostEqual(az_a, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 19994492, delta=0.5)
        s_ab, az_a, az_b = geod.inverse(0, 0, 0, 180, degrees=True)
        self.assertAlmostEqual(az_a, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 90.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 20106193, delta=0.5)
        s_ab, az_a, az_b = geod.inverse(0, 0, 0.5, 180, degrees=True)
        self.assertAlmostEqual(az_a, 33.02493, delta=0.5e-5)
        self.assertAlmostEqual(az_b, 146.97364, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 20082617, delta=0.5)
        s_ab, az_a, az_b = geod.inverse(0, 0, 1, 180, degrees=True)
        self.assertAlmostEqual(az_a, 0.00000, delta=0.5e-5)
        self.assertAlmostEqual(az_b, -180.00000, delta=0.5e-5)
        self.assertAlmostEqual(s_ab, 20027270, delta=0.5)

    def test_GeodSolve55(self):
        # Check fix for nan + point on equator or pole not returning all nans
        # Geodesic::Inverse, found 2015-09-23.
        s_ab, az_a, az_b = wgs84.inverse(np.nan, 0, 0, 90, degrees=True)
        self.assertTrue(np.isnan(az_a))
        self.assertTrue(np.isnan(az_b))
        self.assertTrue(np.isnan(s_ab))
        s_ab, az_a, az_b = wgs84.inverse(np.nan, 0, 90, 9, degrees=True)
        self.assertTrue(np.isnan(az_a))
        self.assertTrue(np.isnan(az_b))
        self.assertTrue(np.isnan(s_ab))


if __name__ == "__main__":
    # import syssys.argv = ['', 'Test.testName']
    unittest.main()
