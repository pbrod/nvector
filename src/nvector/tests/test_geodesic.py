import pytest
from pytest import approx

import numpy as np
from nvector import GeoPoint, FrameE

WGS84 = FrameE(name='WGS84')
SPHERE = FrameE(6.4e6, 0, name='sphere')
PROLATE15 = FrameE(6.4e6, -1 / 150.0, name='prolate spheroid')
PROLATE30 = FrameE(6.4e6, -1 / 300.0, name='prolate spheroid')


WGS84_TESTCASES = [
    # lat1, lon1, azi1, lat2, lon2, azi2, s12
    #(0.01777745589997, 30., 0., 90., 210., 0., 10e6),
    (40.6, -73.8, 53.47021823943233, 49.01666667, 2.55, 111.5936695140232, 5853226.255613291),
    (0, 539, 90, 0, 541, 90, 222638.9815865333),
    (35.60777, -139.44815, 111.09874842956033, -11.17491, -69.95921, 129.28927088970877,
     8935244.56048183),
    (55.52454, 106.05087, 22.0200598809828, 77.03196, 197.18234, 109.11204111067151,
     4105086.171392441),
    (-21.97856, 142.59065, -32.44456876433189, 41.84138, 98.56635, -41.84359951440466,
     8394328.894657671),
    (-66.99028, 112.2363, 173.73491240878403, -12.70631, 285.90344, 2.512956620913668,
     11150344.231208025),
    (-17.42761, 173.34268, -159.03355766119293, -15.84784, 5.93557, -20.78748465153699,
     16076603.163118068),
    (32.84994, 48.28919, 150.492927788122, -56.28556, 202.29132, 48.11344939981676,
     16727068.943816446),
    (6.96833, 52.74123, 92.58158538631771, -7.39675, 206.17291, 90.72169216592391,
     17102477.249695837),
    (-50.56724, -16.30485, -105.43967990759016, -33.56571, -94.97412, -47.34854783565033,
     6455670.511866869),
    (-58.93002, -8.90775, 140.96539790250068, -8.91104, 133.13503, 19.255429433416598,
     11756066.021986462),
    (-68.82867, -74.28391, 93.77434776311487, -50.63005, -8.36685, 34.65564085411343,
     3956936.926063544),
    (-10.62672, -32.0898, -86.42671328674776, 5.883, -134.31681, -80.47378097103487,
     11470869.386456301),
    (-21.76221, 166.90563, 29.31942120693643, 48.72884, 213.97627, 43.50867194641017,
     9098627.398655491),
    (-19.79938, -174.47484, 71.16727578017154, -11.99349, -154.35109, 65.58909977519923,
     2319004.860116939),
    (-11.95887, -116.94513, 92.71261983045255, 4.57352, 7.16501, 78.64960934409585,
     13834722.580140138),
    (-87.85331, 85.66836, -65.12031304024275, 66.48646, 16.09921, -4.888658719272296,
     17286615.314714465),
    (1.74708, 128.32011, -101.58484363117385, -11.16617, 11.87109, -86.32579329643748,
     12942901.124134742),
    (-25.72959, -144.90758, -153.6474686931172, -57.70581, -269.17879, -48.34398315887649,
     9413446.745245311),
    (-41.22777, 122.32875, 14.285113402275739, -7.57291, 130.37946, 10.805303085187369,
     3812686.035106021),
    (11.01307, 138.25278, 79.43682622782374, 6.62726, 247.05981, 103.70809021552266,
     11911190.819018409),
    (-29.47124, 95.14681, -163.77913044168838, -27.46601, -69.15955, -15.90933594555497,
     13487015.838114548),
]
PROLATE15_TESTCASES = [
    (35.60777, -139.44815, 111.61226574744089, -11.17491, -69.95921, 129.74259016618828,
     9013328.507759787),
    (55.52454, 106.05087, 21.965965963735737, 77.03196, 197.18234, 109.05271248950473,
        4090377.934609745),
    (-21.97856, 142.59065, -32.007257221949374, 41.84138, 98.56635, -41.38281215823377,
        8519127.910436656),
    (-66.99028, 112.2363, 173.70200220073562, -12.70631, 285.90344, 2.505987937730224,
        11189532.057805229),
    (-17.42761, 173.34268, -158.4343239089579, -15.84784, 5.93557, -21.377697964630123,
        16153622.691210287),
    (32.84994, 48.28919, 150.19964051382638, -56.28556, 202.29132, 48.954328603130776,
        16862830.23759952),
    (6.96833, 52.74123, 92.65452010474647, -7.39675, 206.17291, 90.88360428961529,
        17163247.367526505),
    (-50.56724, -16.30485, -105.26223530321928, -33.56571, -94.97412, -47.22069554513815,
        6455504.846661402),
    (-58.93002, -8.90775, 140.7065961470972, -8.91104, 133.13503, 19.224255433238767,
        11802358.81901205),
    (-68.82867, -74.28391, 93.66439521205321, -50.63005, -8.36685, 34.55320456856311,
        3946879.8062087535),
    (-10.62672, -32.0898, -86.21939301195984, 5.883, -134.31681, -80.31521657879257,
        11515141.309744174),
    (-21.76221, 166.90563, 28.92965265095955, 48.72884, 213.97627, 43.080582576605686,
        9230015.662360186),
    (-19.79938, -174.47484, 70.79890546568507, -11.99349, -154.35109, 65.22187529367612,
        2331240.8795731547),
    (-11.95887, -116.94513, 92.44932080291142, 4.57352, 7.16501, 78.60406662111117,
        13885902.145201873),
    (-87.85331, 85.66836, -65.06901600981988, 66.48646, 16.09921, -4.878918676309661,
        17469436.386037808),
    (1.74708, 128.32011, -101.62042835889467, -11.16617, 11.87109, -86.54609333914628,
        12989405.48843234),
    (-25.72959, -144.90758, -153.71214867831526, -57.70581, -269.17879, -48.53670031604997,
        9422911.753563497),
    (-41.22777, 122.32875, 14.09884489970991, -7.57291, 130.37946, 10.620664014496121,
        3878224.372310634),
    (11.01307, 138.25278, 79.59110525655268, 6.62726, 247.05981, 103.64549774396255,
        11950217.515074987),
    (-29.47124, 95.14681, -163.6007367991418, -27.46601, -69.15955, -16.07926182285449,
        13519526.601036258),
]

PROLATE30_TESTCASES = [
    (35.60777, -139.44815, 111.44181270308195, -11.17491, -69.95921, 129.59200129071922,
     8997438.73865191),
    (55.52454, 106.05087, 21.983815269254553, 77.03196, 197.18234, 109.07228880412546,
        4099915.9769760696),
    (-21.97856, 142.59065, -32.151801028461286, 41.84138, 98.56635, -41.5351384309475,
        8487080.163153801),
    (-66.99028, 112.2363, 173.712936378524, -12.70631, 285.90344, 2.5082999901065395,
        11189187.599241104),
    (-17.42761, 173.34268, -158.6364532079238, -15.84784, 5.93557, -21.178626464145097,
        16146414.41428386),
    (32.84994, 48.28919, 150.29652996004262, -56.28556, 202.29132, 48.67422547253897,
        16836677.808609143),
    (6.96833, 52.74123, 92.6295362350333, -7.39675, 206.17291, 90.83032731107224,
        17162524.812147524),
    (-50.56724, -16.30485, -105.32104275285913, -33.56571, -94.97412, -47.263142702607645,
        6462913.803563158),
    (-58.93002, -8.90775, 140.79261894285, -8.91104, 133.13503, 19.23468529401296,
        11800337.403053956),
    (-68.82867, -74.28391, 93.7007080270391, -50.63005, -8.36685, 34.58703780610325,
        3954715.7455309834),
    (-10.62672, -32.0898, -86.28839925610205, 5.883, -134.31681, -80.36825950319547,
        11513476.171529751),
    (-21.76221, 166.90563, 29.0584413692963, 48.72884, 213.97627, 43.22207400111539,
        9196594.154815871),
    (-19.79938, -174.47484, 70.92152583788754, -11.99349, -154.35109, 65.34411766319404,
        2329801.7977771307),
    (-11.95887, -116.94513, 92.53629015913596, 4.57352, 7.16501, 78.6201082563427,
        13884640.847192958),
    (-87.85331, 85.66836, -65.08600286278707, 66.48646, 16.09921, -4.882131404284117,
        17428338.398526873),
    (1.74708, 128.32011, -101.60790215008396, -11.16617, 11.87109, -86.47325525979215,
        12988687.410732202),
    (-25.72959, -144.90758, -153.69062835797547, -57.70581, -269.17879, -48.47279845311159,
        9430487.774605595),
    (-41.22777, 122.32875, 14.160238288055323, -7.57291, 130.37946, 10.681515281995829,
        3860767.0683828103),
    (11.01307, 138.25278, 79.54030820948694, 6.62726, 247.05981, 103.66570160682224,
        11950822.587033758),
    (-29.47124, 95.14681, -163.66033624815333, -27.46601, -69.15955, -16.02249795508613,
        13524135.612036409),
]


@pytest.mark.parametrize("testcase", WGS84_TESTCASES)
def test_wgs84_inverse(testcase):
    options = dict(frame=WGS84, degrees=True)

    (lat1, lon1, azi1, lat2, lon2, azi2, s12) = testcase[:7]
    point1 = GeoPoint(lat1, lon1, **options)
    point2 = GeoPoint(lat2, lon2, **options)
    s_ab, az_a, az_b = point1.distance_and_azimuth(point2,
                                                   degrees=True)
    assert s_ab == approx(s12, rel=1e-14)
    assert az_a == approx(azi1, abs=2e-13)
    assert az_b == approx(azi2, abs=1e-13)


@pytest.mark.parametrize("testcase", WGS84_TESTCASES)
def test_wgs84_direct(testcase):
    options = dict(frame=WGS84, degrees=True)

    (lat1, lon1, azi1, lat2, lon2, azi2, s12) = testcase[:7]
    point1 = GeoPoint(lat1, lon1, **options)
    point2, az_b = point1.displace(s12, azi1, long_unroll=True, degrees=True)

    lat_b, lon_b = point2.latitude_deg, point2.longitude_deg

    assert lat_b == approx(lat2, abs=1e-13)
    assert lon_b == approx(lon2, abs=1e-13)
    assert az_b == approx(azi2, abs=1e-13)


@pytest.mark.parametrize("testcase", PROLATE15_TESTCASES)
def test_prolate15_direct(testcase):
    options = dict(frame=PROLATE15, degrees=True)

    (lat1, lon1, azi1, lat2, lon2, azi2, s12) = testcase[:7]
    point1 = GeoPoint(lat1, lon1, **options)
    point2, az_b = point1.displace(s12, azi1, long_unroll=True, degrees=True)

    lat_b, lon_b = point2.latitude_deg, point2.longitude_deg

    assert lat_b == approx(lat2, abs=1e-13)
    assert lon_b == approx(lon2, abs=2e-13)
    assert az_b == approx(azi2, abs=1e-13)


@pytest.mark.parametrize("testcase", PROLATE30_TESTCASES)
def test_prolate30_direct(testcase):
    options = dict(frame=PROLATE30, degrees=True)

    (lat1, lon1, azi1, lat2, lon2, azi2, s12) = testcase[:7]
    point1 = GeoPoint(lat1, lon1, **options)
    point2, az_b = point1.displace(s12, azi1, long_unroll=True, degrees=True)

    lat_b, lon_b = point2.latitude_deg, point2.longitude_deg

    assert lat_b == approx(lat2, abs=1e-13)
    assert lon_b == approx(lon2, abs=1e-13)
    assert az_b == approx(azi2, abs=1e-13)


@pytest.mark.parametrize("lat1, lon1, lat2, lon2, s12, az1, az2",
                         [(0.07476, 0, -0.07476, 180, 20106193, 90.00078, 90.00078),
                          (0.1, 0, -0.1, 180, 20106193, 90.00105, 90.00105)])
def test_geo_solve2(lat1, lon1, lat2, lon2, s12, az1, az2):
    """Check fix for antipodal prolate bug found 2010-09-04"""

    s_ab, az_a, az_b = PROLATE15.inverse(lat1, lon1, lat2, lon2, degrees=True)

    assert az_a == approx(az2, abs=0.5e-5)
    assert az_b == approx(az1, abs=0.5e-5)
    assert s_ab == approx(s12, abs=0.5)


def test_geo_solve4():
    """Check fix for short line bug found 2010-05-21"""
    lat1, lon1, lat2, lon2 = np.deg2rad((36.493349428792, 0,
                                         36.49334942879201, .0000008))
    s_ab, az_a, az_b = WGS84.inverse(lat1, lon1, lat2, lon2)

    assert s_ab == approx(0.072, abs=0.5e-3)
    assert az_a == approx(1.5707963218661298)
    assert az_b == approx(1.5707963218661298)


def test_geo_solve5():
    # Check fix for point2=pole bug found 2010-05-03
    # (0.01777745589997, 30., 0., 90., 210., 0., 10e6),
    lat1, lon1, az1 = (0.01777745589997, 30, 0)
    lat_b, lon_b, az_b = WGS84.direct(lat1, lon1, az1, 1.000000000000001e7, long_unroll=False, degrees=True)

    assert lat_b == approx(90, abs=0.5e-5)
    if lon_b < 0: # just passed the north pole
        assert lon_b == approx(-150, abs=0.5e-5)
        assert az_b == approx(180, abs=0.5e-5)
    else:  # Just stopped before the north pole
        assert lon_b == approx(30, abs=0.5e-5)
        assert az_b == approx(0, abs=0.5e-5)


def test_geo_solve6():
    # Check fix for volatile sbet12a bug found 2011-06-25 (gcc 4.4.4
    # x86 -O3).  Found again on 2012-03-27 with tdm-mingw32 (g++ 4.6.1).

    lat1, lon1, lat2, lon2 = np.deg2rad((88.202499451857, 0,
                                         -88.202499451857,
                                         179.981022032992859592))
    s_ab, _az_a, _az_b = WGS84.inverse(lat1, lon1, lat2, lon2)
    assert s_ab == approx(20003898.214, abs=0.5e-3)

    lat1, lon1, lat2, lon2 = np.deg2rad((89.262080389218, 0,
                                         -89.262080389218,
                                         179.992207982775375662))
    s_ab, _az_a, _az_b = WGS84.inverse(lat1, lon1, lat2, lon2)
    assert s_ab == approx(20003925.854, abs=0.5e-3)

    lat1, lon1, lat2, lon2 = np.deg2rad((89.333123580033, 0,
                                         -89.333123580032997687,
                                         179.99295812360148422))
    s_ab, _az_a, _az_b = WGS84.inverse(lat1, lon1, lat2, lon2)
    assert s_ab == approx(20003926.881, abs=0.5e-3)


def test_geo_solve9():
    # Check fix for volatile x bug found 2011-06-25 (gcc 4.4.4 x86 -O3)
    lat1, lon1, lat2, lon2 = np.deg2rad((56.320923501171, 0,
                                         -56.320923501171,
                                         179.664747671772880215))
    s_ab, _az_a, _az_b = WGS84.inverse(lat1, lon1, lat2, lon2)
    assert s_ab == approx(19993558.287, abs=0.5e-3)


def test_geo_solve10():
    # Check fix for adjust tol1_ bug found 2011-06-25 (Visual Studio
    # 10 rel + debug)
    lat1, lon1, lat2, lon2 = (52.784459512564, 0, -52.784459512563990912, 179.634407464943777557)
    s_ab, _az_a, _az_b = WGS84.inverse(lat1, lon1, lat2, lon2, degrees=True)
    assert s_ab == approx(19991596.095, abs=0.5e-3)


def test_geo_solve11():
    # Check fix for bet2 = -bet1 bug found 2011-06-25 (Visual Studio
    # 10 rel + debug)
    lat1, lon1, lat2, lon2 = (48.522876735459, 0, -48.52287673545898293, 179.599720456223079643)
    s_ab, _az_a, _az_b = WGS84.inverse(lat1, lon1, lat2, lon2, degrees=True)
    assert s_ab == approx(19989144.774, abs=0.5e-3)


def test_geo_solve12():
    # Check fix for inverse geodesics on extreme prolate/oblate
    # ellipsoids Reported 2012-08-29 Stefan Guenther
    # <stefan.gunther@embl.de>; fixed 2012-10-07
    prolate_geod = FrameE(89.8, -1.83)
    lat1, lon1, lat2, lon2 = (0, 0, -10, 160)
    s_ab, az_a, az_b = prolate_geod.inverse(lat1, lon1, lat2, lon2, degrees=True)
    assert az_a == approx(120.27, abs=1e-2)
    assert az_b == approx(105.15, abs=1e-2)
    assert s_ab == approx(266.7, abs=1e-1)


def test_wgs84direct_long_unroll():
    """Check wgs84direct with and without long unroll"""
    lat_b, lon_b, az_b = WGS84.direct(40, -75, -10, 2e7, long_unroll=True, degrees=True)

    assert lat_b == approx(-39, abs=1)
    assert lon_b == approx(-254, abs=1)
    assert az_b == approx(-170, abs=1)

    lat_b, lon_b, az_b = WGS84.direct(40, -75, -10, 2e7, degrees=True)
    assert lat_b == approx(-39, abs=1)
    assert lon_b == approx(105, abs=1)
    assert az_b == approx(-170, abs=1)


@pytest.mark.parametrize("datum, lat1, lon1, lat2, lon2, s12, az1, az2",
                         [(WGS84, 0, 0, 0, 179, 19926189, 90.00000, 90.00000),
                          (WGS84, 0, 0, 0, 179.5, 19980862, 55.96650, 124.03350),
                          (WGS84, 0, 0, 0, 180, 20003931, 0.00000, 180.00000),
                          (WGS84, 0, 0, 1, 180, 19893357, 0.00000, 180.00000),
                          (SPHERE, 0, 0, 0, 179, 19994492, 90.00000, 90.00000),
                          (SPHERE, 0, 0, 0, 180, 20106193, 0.00000, 180.00000),
                          (SPHERE, 0, 0, 1, 180, 19994492, 0.00000, 180.00000),
                          (PROLATE30, 0, 0, 0, 179, 19994492, 90.00000, 90.00000),
                          (PROLATE30, 0, 0, 0, 180, 20106193, 90.00000, 90.00000),
                          (PROLATE30, 0, 0, 0.5, 180, 20082617, 33.02493, 146.97364),
                          (PROLATE30, 0, 0, 1, 180, 20027270, 0.00000, 180.00000)
                          ])
def test_geo_solve33(datum, lat1, lon1, lat2, lon2, s12, az1, az2):
    # Check max(-0.0,+0.0) issues 2015-08-22 (triggered by bugs in
    # Octave -- sind(-0.0) = +0.0 -- and in some version of Visual
    # Studio -- fmod(-0.0, 360.0) = +0.0.
    s_ab, az_a, az_b = datum.inverse(lat1, lon1, lat2, lon2, degrees=True)
    assert az_a == approx(az1, abs=0.5e-5)
    assert az_b == approx(az2, abs=0.5e-5)
    assert s_ab == approx(s12, abs=0.5)


@pytest.mark.parametrize("lat1, lon1, lat2, lon2",
                         [(np.nan, 0, 0, 90),
                          (np.nan, 0, 90, 9),
                          (0, np.nan, 0, 90),
                          (0, 0, np.nan, 90),
                          (0, 0, 1, np.nan)])
def test_nan_propagation(lat1, lon1, lat2, lon2):
    """Check that WGS84.inverse return nans when input is nan."""
    s_ab, az_a, az_b = WGS84.inverse(0, 0, 1, np.nan, degrees=True)
    assert np.isnan(az_a)
    assert np.isnan(az_b)
    assert np.isnan(s_ab)


# def test_make_testcases():
#     options = dict(frame=PROLATE30, degrees=True)
#     for testcase in WGS84_TESTCASES:
#         (lat1, lon1, azi1, lat2, lon2, azi2, s12) = testcase[:7]
#         #print(lat1, lon1, azi1, lat2, lon2, azi2, s12)
#         #print(testcase[:7], ',')
#         point1 = GeoPoint(lat1, lon1, **options)
#         point2 = GeoPoint(lat2, lon2, **options)
#         s_ab, az_a, az_b = point1.distance_and_azimuth(point2,
#                                                        long_unroll=True,
#                                                        degrees=True)
#         new_testcase = (lat1, lon1, az_a, lat2, lon2, az_b, s_ab)
#         print(new_testcase, ',')
#     assert False
