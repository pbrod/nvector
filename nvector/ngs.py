'''
Vincenty's formulae are two related iterative methods used in geodesy to
calculate the distance between two points on the surface of a spheroid,
developed by Thaddeus Vincenty (1975a) They are based on the assumption that
the figure of the Earth is an oblate spheroid, and hence are more accurate than
methods such as great-circle distance which assume a spherical Earth.
The first (direct) method computes the location of a point which is a given
distance and azimuth (direction) from another point. The second (inverse)
method computes the geographical distance and azimuth between two given points.
They have been widely used in geodesy because they are accurate to within
0.5 mm (0.020'') on the Earth ellipsoid.


Reference
---------


 name:      inverse
 version:   201105.xx
 author:    stephen j. frakes
 last mod:  dr. dennis milbert
 purpose:   to compute a geodetic inverse
            and then display output information


 local variables and constants:
 ------------------------------
 a                semimajor axis equatorial (in meters)
 f                flattening
 b                semiminor axis polar (in meters)
 baz              azimuth back (in radians)

 dlon             temporary value for difference in longitude (radians)

 edist            ellipsoid distance (in meters)
 elips            ellipsoid choice

 faz              azimuth forward (in radians)

 finv             reciprocal flattening

 option           user prompt response

 name1            name of station one
 glat1,glon1      station one       - (lat & lon in radians )

 name2            name of station two
 glat2,glon2      station two       - (lat & lon in radians )


'''
from __future__ import division, print_function
import numpy as np
from numpy import pi, arctan, arctan2, tan, sin, cos, sqrt
from nvector import lat_lon2n_E, azimuth

_EPS = np.finfo(float).eps

ELLIPSOID = {1: (6377563.3960, 1.0/299.3249646, 'Airy 1858'),
             2: (6377340.189, 1.0/299.3249646, 'Airy Modified'),
             3: (6378160, 1.0/298.25, 'Australian National'),
             4: (6377397.155, 1.0/299.1528128, 'Bessel 1841'),
             5: (6378249.145, 1.0/293.465, 'Clarke 1880'),
             6: (6377276.345, 1.0/300.8017, 'Everest 1830'),
             7: (6377304.063, 1.0/300.8017, 'Everest Modified'),
             8: (6378166.0, 1.0/298.3, 'Fisher 1960'),
             9: (6378150.0, 1.0/298.3, 'Fisher 1968'),
             10: (6378270.0, 1.0/297, 'Hough 1956'),
             11: (6378388.0, 1.0/297, 'International (Hayford)'),
             12: (6378245.0, 1.0/298.3, 'Krassovsky 1938'),
             13: (6378145., 1.0/298.25, 'NWL-9D  (WGS 66)'),
             14: (6378160., 1.0/298.25, 'South American 1969'),
             15: (6378136, 1.0/298.257, 'Soviet Geod. System 1985'),
             16: (6378135., 1.0/298.26, 'WGS 72'),
             17: (6378206.4, 1.0/294.9786982138, 'Clarke 1866    (NAD27)'),
             18: (6378137.0, 1.0/1.0/298.257223563,
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
    17) User defined.
    '
    Enter choice :
    """
    if name:
        option = ELLIPSOID_IX.get(name.lower().replace(' ', ''), name)
    else:
        option = input(msg)
    return ELLIPSOID[option]

#       else
#         elips = 'User defined.'
#
#         write(*,*) '  Enter Equatorial axis,   a : '
#         read(*,*) a
#         a  = dabs(a)
#
#         write(*,*) '  Enter either Polar axis, b or '
#         write(*,*) '  Reciprocal flattening,   1/f : '
#         read(*,*) ss
#         ss = dabs(ss)
#
#         f = 0.0d0
#         if( 200.0d0.le.ss .and. ss.le.310.0d0 )then
#           f = 1.d0/ss
#         elseif( 6000000.0d0<ss .and. ss<a )then
#           f = (a-ss)/a
#         else
#           elips = 'Error: default GRS80 used.'
#           a     = 6378137.0d0
#           f     = 1.0d0/298.25722210088d0
#         endif
#       endif
#
#       return
#       end


def _compute_c(rf, cosal2):
    return ((4.0 - 3.0 * cosal2) / rf + 4.0) * cosal2 / rf / 16.0  # Eq. 10


def _costm(sinu1sinu2, cosal2, cossig):
    sgn = np.where(cosal2 >= 0.0, 1, -1)
    return -2.0 * (sinu1sinu2 / np.where(np.abs(cosal2) < _EPS,
                                         _EPS * sgn,
                                        cosal2)) + cossig  # avoid div 0


def _cossinsig(cosu1, sinu1, cosu2, sinu2, coslam, sinlam):
    temp = cosu1 * sinu2 - sinu1 * cosu2 * coslam
    sinsig = sqrt((cosu2 * sinlam) ** 2 + temp ** 2)  # v14   (rapp part II)
    cossig = sinu1*sinu2 + cosu1*cosu2*coslam         # v15   (rapp part II)
    return cossig, sinsig


def _big_ab(u2):
    bige = sqrt(1.0 + u2)                            # 15
    bigf = (bige - 1.0) / (bige + 1.0)               # 16
    biga = (1.0 + bigf * bigf / 4.0) / (1.0 - bigf)  # 17
    bigb = bigf * (1.0 - 0.3750 * bigf * bigf)       # 18
    return biga, bigb


def _dsigma(bigb, cossig, sinsig, costm):
    costm2 = costm**2
    z = bigb / 6.0 * costm * (4.0 * sinsig**2 - 3.0) * (4.0*costm2 - 3.0)
    dsig = bigb*sinsig*(costm + bigb/4.0 * (cossig * (2.0*costm2 - 1.0) - z))
    return dsig  # 19


def _helmert(a, boa, cossig, sinsig, sig, cosal2, costm):
    """Helmert (1880) from Vincenty
    'Geodetic inverse solution between antipodal points'
    """
    u2 = (1.0 / (boa * boa) - 1.0) * cosal2
    biga, bigb = _big_ab(u2)
    dsig = _dsigma(bigb, cossig, sinsig, costm)
    s12 = (boa * a) * biga * np.abs(sig - dsig)  # 20
    return s12


def _antipodal_azimuths(cosu1, sinu1, cosu2, sinu2, coslam, sinal,
                       cossig, sinsig):
    faz = sinal / cosu1
    sgn = np.where(cosu1 * sinu2 - sinu1 * cosu2 * coslam < 0, -1, 1)
    baz = sqrt(1.0 - faz**2) * sgn
    faz = arctan2(faz, baz)
    baz = arctan2(-sinal, sinu1 * sinsig - cosu1 * cossig * baz)
    return faz, baz


def _long_line_azimuths(cosu1, sinu1, cosu2, sinu2, coslam, sinlam):
    faz = arctan2(cosu2 * sinlam, cosu1 * sinu2 - sinu1 * cosu2 * coslam)
    baz = arctan2(-cosu1 * sinlam, sinu1 * cosu2 - cosu1 * sinu2 * coslam)
    return faz, baz


def _refined_converge(val, test, prev, it):
    if(((val-test)*(test-prev)) < 0.0 and it > 5):
        val = (2.0*val+3.0*test+prev)/6.0      # refined converge.
    return val, val, test


def _diff_longitude(l1, l2):
    l = l2 - l1  # longitude difference [-pi,pi]
    if (l > pi):
        l = l - pi - pi
    if (l < -pi):
        l = l + pi + pi
    return l


def _cossinal(cosu1, cosu2, sinsig, sinlam):
    sgn = np.where(sinsig < 0, -1, 1)
    sinal = cosu1 * cosu2 * sinlam / np.where(np.abs(sinsig) < _EPS,
                                              _EPS * sgn,
                                              sinsig)  # avoid div 0
    cosal2 = 1.0 - sinal ** 2
    return cosal2, sinal


def _reduced_latitude(lat1, boa):
    u1 = arctan(boa * tan(lat1))  # better reduced latitude
    cosu1, sinu1 = cos(u1), sin(u1)
    return cosu1, sinu1


class Geodesic(object):
    """Solve geodesic problems.

    The following illustrates its use

    >>> import numpy as np
    >>> from ngs import Geodesic

    >>> wgs84 = Geodesic(name='WGS84')

    # The geodesic inverse problem
    >>> lat1, lon1 = np.deg2rad((-41.32, 174.81))
    >>> lat2, lon2 = np.deg2rad((40.96, -5.50))
    >>> s12, az1, az2 = wgs84.inverse(lat1, lon1, lat2, lon2)[:3]

    # The geodesic direct problem
    >>> lat1, lon1, az1 = np.deg2rad((40.6, -73.8, 45))
    >>> lat2, lon2, az2 = wgs84.direct(lat1, lon1, az1, 10000e3)

    All angles (latitudes, longitudes, azimuths, spherical arc lengths)
    are measured in radians.  Latitudes must lie in [-pi/2,pi/2].  All lengths
    (distance, reduced length) are measured in meters.

    """
    def __init__(self, a=6378137, f=1.0/298.257223563, name=''):
        if name:
            a, f, _full_name = select_ellipsoid(name)
        self.a = a
        self.f = f
        self.name = name

    def inverse(self, lat1, lon1, lat2, lon2):
        """
        Return ellipsoidal distance between points
        inverse for long-line and antipodal cases.
        latitudes may be 90 degrees exactly.
        latitude positive north, longitude positive east, radians.
        azimuth clockwise from north, radians.
        original programmed by thaddeus vincenty, 1975, 1976
        removed back side solution option,
        debugged, revised -- 2011may01 -- dgm
        this version of code is interim -- antipodal boundary needs work

        Returns
        -------
        s12: real scalar
            ellopsoidal distance between point 1 and 2
        faz, baz: real scalars
            forward and backward azimuth


        sig,  spherical distance on auxiliary sphere
        lam,  longitude difference on auxiliary sphere
        kind, solution flag:  kind=1, long-line;  kind=2, antipodal
        it,   iteration count
        """

        tol = 1.e-14
        if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0
        n_EA_E = lat_lon2n_E(lat1, lon1)
        n_EB_E = lat_lon2n_E(lat2, lon2)
        faz1 = azimuth(n_EA_E, n_EB_E, self.a, self.f)
        baz1 = azimuth(n_EB_E, n_EA_E, self.a, self.f)
        if lon1 < 0.0:
            lon1 = lon1+2.0*pi

        if lon2 < 0.0:
            lon2 = lon2+2.0*pi

        a, rf = self.a, 1.0/self.f
        boa = 1.0 - 1.0/rf
        cosu1, sinu1 = _reduced_latitude(lat1, boa)
        cosu2, sinu2 = _reduced_latitude(lat2, boa)

        lon_diff = _diff_longitude(lon1, lon2)
        prev, test = lon_diff, lon_diff
        it, kind, lam = 0, 1, lon_diff                  # v13   (rapp part II)

        stop_running = False
        # top of the long-line loop (kind=1)
        while not stop_running:  # 2
            coslam, sinlam = cos(lam), sin(lam)
            cossig, sinsig = _cossinsig(cosu1, sinu1, cosu2, sinu2, coslam, sinlam)
            sig = arctan2(sinsig, cossig)
            cosal2, sinal = _cossinal(cosu1, cosu2, sinsig, sinlam)
            costm = _costm(sinu1*sinu2, cosal2, cossig)

            c = _compute_c(rf, cosal2)    # v10   (rapp part II)

            # entry point of the antipodal loop (kind=2)
            while True:  # 6
                it = it+1
                # v11
                d = (((2.0*costm**2-1.0)*cossig*c+costm)*sinsig*c+sig)*(1.0-c)/rf
                if kind == 1:
                    lam = lon_diff+d*sinal
                    stop_running = (np.abs(lam-test) < tol)
                    if stop_running:
                        break  # go to 100

                    if(np.abs(lam) > pi):
                        kind = 2
                        lam = pi
                        if(lon_diff < 0.0):
                            lam = -lam
                        sinal, cosal2 = 0.0, 1.0
                        prev, test = 2.0, 2.0
                        sig = pi - np.abs(arctan(sinu1/cosu1) + arctan(sinu2/cosu2))
                        sinsig, cossig = sin(sig), cos(sig)
                        c = _compute_c(rf, cosal2)
                        stop_running = (np.abs(sinal-prev) < tol)
                        if stop_running:
                            break  # go to 100
                        costm = _costm(sinu1*sinu2, cosal2, cossig)
                        continue  # go to 6
                        # endif
                    lam, test, prev = _refined_converge(lam, test, prev, it)
                    break  # go to 2
                else:
                    sinal = (lam-lon_diff)/d
                    sinal, test, prev = _refined_converge(sinal, test, prev, it)

                    cosal2 = 1.0 - sinal**2
                    sinlam = sinal*sinsig/(cosu1*cosu2)
                    coslam = -sqrt(np.abs(1.0 - sinlam**2))
                    lam = arctan2(sinlam, coslam)
                    cossig, sinsig = _cossinsig(cosu1, sinu1, cosu2, sinu2,
                                                coslam, sinlam)

                    sig = arctan2(sinsig, cossig)
                    c = _compute_c(rf, cosal2)
                    stop_running = (np.abs(sinal-prev) < tol)
                    if stop_running:
                        break  # go to 100

                    costm = _costm(sinu1*sinu2, cosal2, cossig)
                    # go to 6
                # endif
    # convergence

    #  100
        if(kind == 2):    # antipodal
            faz, baz = _antipodal_azimuths(cosu1, sinu1, cosu2, sinu2,
                                          coslam, sinal, cossig, sinsig,)
        else:                                      # long-line
            faz, baz = _long_line_azimuths(cosu1, sinu1, cosu2, sinu2,
                                          coslam, sinlam)
#         if(faz < 0.0):
#             faz = faz+pi+pi
#         if(baz < 0.0):
#             baz = baz+pi+pi


        s12 = _helmert(a, boa, cossig, sinsig, sig, cosal2, costm)

        return s12, faz, baz, sig, lam, kind, it
        # end

    def direct(self, lat1, lon1, faz, S):
        """
        SOLUTION OF THE GEODETIC DIRECT PROBLEM AFTER T.VINCENTY
        MODIFIED RAINSFORD'S METHOD WITH HELMERT'S ELLIPTICAL TERMS
        EFFECTIVE IN ANY AZIMUTH AND AT ANY DISTANCE SHORT OF ANTIPODAL

        A IS THE SEMI-MAJOR AXIS OF THE REFERENCE ELLIPSOID
        F IS THE FLATTENING OF THE REFERENCE ELLIPSOID
        LATITUDES AND LONGITUDES IN RADIANS POSITIVE NORTH AND EAST
        AZIMUTHS IN RADIANS CLOCKWISE FROM NORTH
        GEODESIC DISTANCE S ASSUMED IN UNITS OF SEMI-MAJOR AXIS A

        PROGRAMMED FOR CDC-6600 BY LCDR L.PFEIFER NGS ROCKVILLE MD 20FEB75
        MODIFIED FOR SYSTEM 360 BY JOHN G GERGEN NGS ROCKVILLE MD 750608

        """
        a, f = self.a, self.f
        EPS = 0.5e-13
        R = 1.0 - f
        TU = R * sin(lat1)/cos(lat1)
        SF = sin(faz)
        CF = cos(faz)
        baz = 0.0
        if (CF != 0.0):
            baz = arctan2(TU, CF)*2.0
        CU = 1./sqrt(TU*TU+1.)
        SU = TU*CU
        SA = CU*SF
        C2A = -SA*SA+1.0
        X = sqrt((1./R/R-1.)*C2A+1.)+1.
        X = (X-2.)/X
        C = 1.-X
        C = (X*X/4.+1)/C
        D = (0.375*X*X-1.)*X
        TU = S/R/a/C
        Y = TU
        while (np.abs(Y-C) > EPS):  # 100
            SY, CY = sin(Y), cos(Y)
            CZ = cos(baz+Y)
            E = CZ*CZ*2.0-1.0
            C = Y
            X = E*CY
            Y = E + E - 1.0
            Y = (((SY*SY*4.-3.)*Y*CZ*D/6.+X)*D/4.-CZ)*SY*D+TU
            # IF(np.abs(Y-C).GT.EPS)GO TO 100

        baz = CU*CY*CF-SU*SY
        C = R*sqrt(SA*SA + baz*baz)
        D = SU*CY+CU*SY*CF
        lat2 = arctan2(D, C)
        C = CU*CY-SU*SY*CF
        X = arctan2(SY*SF, C)
        C = ((-3.*C2A+4.0) * f + 4.) * C2A * f / 16.0
        D = ((E*CY*C+CZ)*SY*C+Y)*SA
        lon2 = lon1 + X - (1.0 - C) * D * f #+ np.pi*2, np.pi*2)-np.pi
        baz = arctan2(SA, baz)+pi
        lon2 = np.mod(lon2+ np.pi, np.pi*2)-np.pi
        return lat2, lon2, baz


wgs84 = Geodesic(name='WGS84')


def main():
    option = 1
    # option = input(
    print("""
    Program Inverse  -  Version 3.0

    Ellipsoid options :

    1) GRS80 / WGS84  (NAD83)
    2) Clarke 1866    (NAD27)
    3) Any other ellipsoid

    Enter choice :
    """)

    if option == 1:
        a = 6378137.0
        f = 1.0/298.257222100882711243162836600094
        elips = 'GRS80 / WGS84  (NAD83)'
    elif (option == 2):
        a = 6378206.4
        f = 1.0/294.9786982138
        elips = 'Clarke 1866    (NAD27)'
    elif (option == 3):
        a, f, elips = select_ellipsoid()
    else:
        print('Not a valid option. Quitting')

    # esq = f*(2.0-f)

    print('  Enter First Station ')
    lat1, lon1 = np.deg2rad(0), np.deg2rad(0)

    print('  Enter Second Station ')
    lat2, lon2 = np.deg2rad(0.5), np.deg2rad(179.5)

#     compute the geodetic inverse
    wgs84 = Geodesic(name='WGS84')
    finv = 1.0/f
    s12, az1, az2, sigma, dlambda, kind, numiter = wgs84.inverse(lat1, lon1,
                                                                 lat2, lon2)
    lat22, lon22, baz2 = wgs84.direct(lat1, lon1, az1, s12)
#     check for a non distance ... lat1,lon1 & lat2,lon2 equal zero ?
    if s12 < 0.00005:
        az1 = 0.0
        az2 = 0.0
    print(elips)
    print('  Ellipsoid : ')
    print('  Equatorial axis,    a   = %15.4f' % a),
    print('  Polar axis,         b   = %15.4f' % (a*(1.0-f)))
    print('  Inverse flattening, 1/f = %16.11f' % finv)
    print('')
    print('  First  Station : ')
    print('  ---------------- ')
    print('    LAT = %2.3f' % np.rad2deg(lat1))
    print('    LON = %2.3f' % np.rad2deg(lon1))
    print('')
    print('')
    print('  Second  Station : ')
    print('  ---------------- ')
    print('    LAT = %2.3f' % np.rad2deg(lat2))
    print('    LON = %2.3f' % np.rad2deg(lon2))
    print('')
    print('Ellipsoidal distance  S = %14.4f [m]' % s12)
    print('Forward azimuth    AZ1 = %2.2f [deg] from North' % np.rad2deg(az1))
    print('Back azimuth       AZ2 = %2.2f [deg] from North' % np.rad2deg(az2))
    print('iter = %d' % numiter)
    19944127.421


if __name__ == '__main__':
    # print(dict((ELLIPSOID[key][2].lower().replace(' ',''),key)for key in ELLIPSOID))

    main()
