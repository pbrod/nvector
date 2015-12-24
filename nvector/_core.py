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
from numpy import (rad2deg, deg2rad, arctan2, sin, cos, array, cross, dot,
                   sqrt, arccos, pi)
from numpy.linalg import norm
from geographiclib.geodesic import Geodesic as _Geodesic
import warnings


NORTH_POLE = dict(z=array([[0, 0, 1],
                           [0, 1, 0],
                           [-1, 0, 0]]),
                  x=np.eye(3))
R_Ee = NORTH_POLE['z']
_EPS = np.finfo(float).eps  # machine precision (machine epsilon)


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
             18: (6378137.0, 1.0/298.257223563,
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


class _BaseFrame(object):
    def __eq__(self, other):
        try:
            if self is other:
                return True
            return self._is_equal_to(other)
        except AttributeError:
            return False
        raise ValueError

    def _is_equal_to(self, other):
        return False


class FrameE(_BaseFrame):
    """
    E frame
    -------
    Name:
        Earth
    Position:
        The origin coincides with Earth's centre (geometrical centre of
        ellipsoid model).
    Orientation:
        The x-axis is along the Earth's rotation axis, pointing north
        (the yz-plane coincides with the equatorial plane), the y-axis points
        towards longitude +90x (east).
    Comments:
        The frame is Earth-fixed (rotates and moves with the Earth). The choice
        of axis directions ensures that at zero latitude and longitude, N
        (described below) has the same orientation as E. If roll/pitch/yaw are
        zero, also B (described below) has this orientation. Note that these
        properties are not valid for another common choice of the axis
        directions, denoted e (lower case), which has z pointing north and x
        pointing to latitude=longitude=0.
    """
    def __init__(self, a=None, f=None, name='WGS84', north='z'):
        if a is None or f is None:
            a, f, _full_name = select_ellipsoid(name)
        self.a = a
        self.f = f
        self.name = name
        self.R_Ee = NORTH_POLE['z']

    def _is_equal_to(self, other):
        return (np.allclose(self.a, other.a) and
                np.allclose(self.f, other.f) and
                np.allclose(self.R_Ee, other.R_Ee))

    def inverse(self, lat_a, lon_a, lat_b, lon_b, z=0, long_unroll=False,
                degrees=False):

        outmask = _Geodesic.STANDARD
        if long_unroll:
            outmask = _Geodesic.STANDARD | _Geodesic.LONG_UNROLL

        geo = _Geodesic(self.a-z, self.f)
        if not degrees:
            lat_a, lon_a, lat_b, lon_b = deg((lat_a, lon_a, lat_b, lon_b))
        result = geo.Inverse(lat_a, lon_a, lat_b, lon_b, outmask=outmask)
        azimuth_a = result['azi1'] if degrees else rad(result['azi1'])
        azimuth_b = result['azi2'] if degrees else rad(result['azi2'])
        return result['s12'], azimuth_a, azimuth_b

    def direct(self, lat_a, lon_a, azimuth, distance, z=0, long_unroll=False,
               degrees=False):
        geo = _Geodesic(self.a-z, self.f)
        outmask = _Geodesic.STANDARD
        if long_unroll:
            outmask = _Geodesic.STANDARD | _Geodesic.LONG_UNROLL
        if not degrees:
            lat_a, lon_a, azimuth = deg((lat_a, lon_a, azimuth))
        result = geo.Direct(lat_a, lon_a, azimuth, distance, outmask=outmask)
        latb, lonb, azimuth_b = result['lat2'], result['lon2'], result['azi2']
        if not degrees:
            return rad(latb), rad(lonb), rad(azimuth_b)
        return latb, lonb, azimuth_b


class FrameB(_BaseFrame):
    """
    B frame
    -------
    Name:
        Body (typically of a vehicle)
    Position:
        The origin is in the vehicle's reference point.
    Orientation:
        The x-axis points forward, the y-axis to the right (starboard) and the
        z-axis in the vehicle's down direction.
    Comments:
        The frame is fixed to the vehicle.
    """
    def __init__(self, nvector, yaw=0, pitch=0, roll=0, degrees=False):
        self.nvector = nvector
        if degrees:
            yaw, pitch, roll = rad(yaw), rad(pitch), rad(roll)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    @property
    def R_EN(self):
        R_NB = zyx2R(self.yaw, self.pitch, self.roll)
        n_EB_E = self.nvector.normal
        R_EN = n_E2R_EN(n_EB_E, self.nvector.frame.R_Ee)
        return np.dot(R_EN, R_NB)  # rotation matrix

    def _is_equal_to(self, other):
        return (np.allclose(self.yaw, other.yaw) and
                np.allclose(self.pitch, other.pitch) and
                np.allclose(self.roll, other.roll) and
                np.allclose(self.R_EN, other.R_EN) and
                self.nvector == other.nvector)


class FrameN(_BaseFrame):
    """
    N frame
    -------
    Name:
        North-East-Down (local level)
    Position:
        The origin is directly beneath or above the vehicle (B), at Earth's
        surface (surface of ellipsoid model).
    Orientation:
        The x-axis points towards north, the y-axis points towards east
        (both are horizontal), and the z-axis is pointing down.
    Comments:
        When moving relative to the Earth, the frame rotates about its z-axis
        to allow the x-axis to always point towards north. When getting close
        to the poles this rotation rate will increase, being infinite at the
        poles. The poles are thus singularities and the direction of the
        x- and y-axes are not defined here. Hence, this coordinate frame is
        NOT SUITABLE for general calculations.
    """
    def __init__(self, nvector):
        n_EA_E = nvector.normal
        self.nvector = Nvector(n_EA_E, z=0, frame=nvector.frame)
        self.R_EN = n_E2R_EN(n_EA_E, nvector.frame.R_Ee)

    def _is_equal_to(self, other):
        return (np.allclose(self.R_EN, other.R_EN) and
                self.nvector == other.nvector)


class FrameL(FrameN):
    """
    L frame
    -------
    Name:
        Local level, Wander azimuth
    Position:
        The origin is directly beneath or above the vehicle (B), at Earth's
        surface (surface of ellipsoid model).
    Orientation:
        The z-axis is pointing down. Initially, the x-axis points towards
        north, and the y-axis points towards east, but as the vehicle moves
        they are not rotating about the z-axis (their angular velocity relative
        to the Earth has zero component along the z-axis).
        (Note: Any initial horizontal direction of the x- and y-axes is valid
        for L, but if the initial position is outside the poles, north and east
        are usually chosen for convenience.)
    Comments:
        The L-frame is equal to the N-frame except for the rotation about the
        z-axis, which is always zero for this frame (relative to E). Hence, at
        a given time, the only difference between the frames is an angle
        between the x-axis of L and the north direction; this angle is called
        the wander azimuth angle. The L-frame is well suited for general
        calculations, as it is non-singular.
    """
    def __init__(self, nvector, wander_azimuth=0):
        n_EA_E = nvector.normal
        R_Ee = nvector.frame.R_Ee
        self.nvector = Nvector(n_EA_E, z=0, frame=nvector.frame)
        self.R_EN = n_E_and_wa2R_EL(n_EA_E, wander_azimuth, R_Ee=R_Ee)


def frame_definitions():
    """
    Coordinate frame definitions
    ----------------------------

    """ + FrameE.__doc__ + FrameB.__doc__ + FrameN.__doc__ + FrameL.__doc__
    pass


class GeoPoint(object):
    """
    Geographical point

    Parameters
    ----------
    latitude, longitude: real scalars or vectors of length n.
        Geodetic latitude and longitude given in [rad or deg]
    z: real scalar or vector of length n.
        Depth(s) [m]  relative to the ellipsoid (depth = -height)
    frame: FrameE object
        reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.
    degrees: bool
        True if input are given in degrees otherwise radians are assumed.

    Examples
    --------
    Solve geodesic problems.

    The following illustrates its use


    >>> import nvector as nv
    >>> options = dict(frame=nv.FrameE(name='WGS84'), degrees=True)

    # The geodesic inverse problem
    >>> point1 = nv.GeoPoint(-41.32, 174.81, **options)
    >>> point2 = nv.GeoPoint(40.96, -5.50, **options)
    >>> s12, az1, az2 = point1.distance_and_azimuth(point2, degrees=True)
    >>> 's12 = {:5.2f}, az1 = {:5.2f}, az2 = {:5.2f}'.format(s12, az1, az2)
    's12 = 19959679.27, az1 = 161.07, az2 = 18.83'

    # The geodesic direct problem
    >>> point1 = nv.GeoPoint(40.6, -73.8, **options)
    >>> az1, distance = 45, 10000e3
    >>> point2, az2 = point1.geo_point(distance, az1, degrees=True)
    >>> lat2, lon2 = point2.latitude_deg, point2.longitude_deg
    >>> 'lat2 = {:5.2f}, lon2 = {:5.2f}, az2 = {:5.2f}'.format(lat2, lon2, az2)
    'lat2 = 32.64, lon2 = 49.01, az2 = 140.37'

    """
    def __init__(self, latitude, longitude, z=0, frame=None, degrees=False):
        if degrees:
            latitude, longitude = rad(latitude), rad(longitude)
        self.latitude = latitude
        self.longitude = longitude
        self.z = z
        if frame is None:
            frame = FrameE()
        self.frame = frame

    @property
    def latitude_deg(self):
        return deg(self.latitude)

    @property
    def longitude_deg(self):
        return deg(self.longitude)

    def to_nvector(self):
        """
        Converts latitude and longitude to n-vector.

        Parameters
        ----------
        latitude, longitude: real scalars or vectors of length n.
            Geodetic latitude and longitude given in [rad]

        Returns
        -------
        n_E: 3 x n array
            n-vector(s) [no unit] decomposed in E.

        See also
        --------
        n_E2lat_lon.
        """
        latitude, longitude = self.latitude, self.longitude
        n_E = lat_lon2n_E(latitude, longitude, self.frame.R_Ee)
        return Nvector(n_E, self.z, self.frame)

    def to_ecef_vector(self):
        return self.to_nvector().to_ecef_vector()

    def geo_point(self, distance, azimuth, long_unroll=False, degrees=False):
        """
        Return position B computed from current position, distance and azimuth.

        Parameters
        ----------
        distance: real scalar
            ellipsoidal distance [m] between position A and B.
        azimuth_a:
            azimuth [rad or deg] of line at position A.
        degrees: bool
            azimuths are given in degrees if True otherwise in radians.

        Returns
        -------
        point_b:  GeoPoint object
            latitude and longitude of position B.
        azimuth_b
            azimuth [rad or deg] of line at position B.

        """
        E = self.frame
        z = self.z
        if not degrees:
            azimuth = deg(azimuth)
        lat_a, lon_a = self.latitude_deg, self.longitude_deg
        latb, lonb, azimuth_b = E.direct(lat_a, lon_a, azimuth, distance, z=z,
                                         long_unroll=long_unroll, degrees=True)
        if not degrees:
            azimuth_b = rad(azimuth_b)
        point_b = GeoPoint(latitude=latb, longitude=lonb, z=z,
                           frame=E, degrees=True)
        return point_b, azimuth_b

    def distance_and_azimuth(self, point, long_unroll=False, degrees=False):
        """
        Return ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        point:  GeoPoint object
            Latitude and longitude of position B.
        degrees: bool
            azimuths are returned in degrees if True otherwise in radians.

        Returns
        -------
        s_ab: real scalar
            ellipsoidal distance [m] between position A and B.
        azimuth_a, azimuth_b
            direction [rad or deg] of line at position A and B relative to
            North, respectively.

        """
        if not self.frame == point.frame:
            raise ValueError('E-frames are note equal')

        lat_a, lon_a = self.latitude, self.longitude
        lat_b, lon_b = point.latitude, point.longitude
        if degrees:
            lat_a, lon_a, lat_b, lon_b = deg((lat_a, lon_a, lat_b, lon_b))
        return self.frame.inverse(lat_a, lon_a, lat_b, lon_b, z=self.z,
                                  long_unroll=long_unroll, degrees=degrees)


class Nvector(object):
    """
    N-vector object

    Parameters
    ----------
    normal: 3 x n array
        n-vector(s) [no unit] decomposed in E.
    z: real scalar or vector of length n.
        Depth(s) [m]  relative to the ellipsoid (depth = -height)
    frame: FrameE object
        reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, n_EB_E and a depth, z relative to the
    ellipsiod.

    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """
    def __init__(self, normal, z=0, frame=None):
        self.normal = normal
        self.z = z
        if frame is None:
            frame = FrameE()
        self.frame = frame

    def to_geo_point(self):
        """
        Converts n-vector to geo-point.

        See also
        --------
        n_E2lat_lon, GeoPoint, ECEFvector, Pvector
        """
        n_E = self.normal
        latitude, longitude = n_E2lat_lon(n_E, R_Ee=self.frame.R_Ee)
        return GeoPoint(latitude, longitude, self.z, self.frame)

    def to_ecef_vector(self):
        """
        Converts n-vector to Cartesian position vector ("ECEF-vector")

        Returns
        -------
        p_EB_E:  ECEFvector object
            Cartesian position vector(s) from E to B, decomposed in E.

        The calculation is excact, taking the ellipsity of the Earth into
        account. It is also non-singular as both n-vector and p-vector are
        non-singular (except for the center of the Earth).

        See also
        --------
        n_EB_E2p_EB_E, ECEFvector, Pvector, GeoPoint
        """
        frame = self.frame
        n_EB_E = self.normal
        a, f, R_Ee = frame.a, frame.f, frame.R_Ee
        p_EB_E = n_EB_E2p_EB_E(n_EB_E, depth=self.z, a=a, f=f, R_Ee=R_Ee)
        return ECEFvector(p_EB_E, self.frame)


def diff_nvectors(n_EA_E, n_EB_E):
    """
    From two positions A and B, finds the delta position.

    Parameters
    ----------
    n_EA_E, n_EB_E: Nvector objects
        n-vector(s) [no unit] of position A and B, decomposed in E.

    Returns
    -------
    p_AB_E:  ECEFvector
        Cartesian position vector(s) from A to B, decomposed in E.

    The n-vectors for positions A (n_EA_E) and B (n_EB_E) are given. The
    output is the delta vector from A to B (p_AB_E).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E.
    """
    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = n_EA_E.to_ecef_vector()
    p_EB_E = n_EB_E.to_ecef_vector()
    p_AB_E = -p_EA_E + p_EB_E
    return p_AB_E


class Pvector(object):
    def __init__(self, pvector, frame):
        self.pvector = pvector
        self.frame = frame

    def to_ecef_vector(self):
        frame_B = self.frame
        p_AB_N = self.pvector
        p_AB_E = np.dot(frame_B.R_EN, p_AB_N)
        return ECEFvector(p_AB_E, frame=frame_B.nvector.frame)

    def to_nvector(self):
        self.to_ecef_vector().to_nvector()

    def to_geo_point(self):
        self.to_ecef_vector().to_geo_point()


class ECEFvector(object):
    """
    ECEF-vector object

    Parameters
    ----------
    pvector: 3 x n array
        Cartesian position vector(s) [m] from E to B, decomposed in E.
    frame: FrameE object
        reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    The position of B (typically body) relative to E (typically Earth) is
    given into this function as p-vector, p_EB_E relative to the center of the
    frame.

    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """
    def __init__(self, pvector, frame=None):
        self.pvector = pvector
        if frame is None:
            frame = FrameE()
        self.frame = frame

    def change_frame(self, frame):
        """
        Converts to Cartesian position vector in another frame

        Parameters
        ----------
        frame: FrameB, FrameN or frameL object
            Frame N used to convert p_AB_E (position vector from A to B,
            decomposed in E) to p_AB_N.

        Returns
        -------
        p_AB_N:  Pvector object
            position vector from A to B, decomposed in frame N.

        See also
        --------
        n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
        """
        if not self.frame == frame.nvector.frame:
            raise ValueError('E-frames are not equal')
        p_AB_E = self.pvector
        p_AB_N = np.dot(frame.R_EN.T, p_AB_E)
        return Pvector(p_AB_N, frame=frame)

    def to_geo_point(self):
        """
        Converts ECEF-vector to geo-point.

        Returns
        -------
        point: GeoPoint object
            containing geodetic latitude and longitude given in [rad or deg]
            and depth, z, relative to the ellipsoid (depth = -height).

        See also
        --------
        n_E2lat_lon, n_EB_E2p_EB_E,  GeoPoint, Nvector, ECEFvector, Pvector
        """
        return self.to_nvector().to_geo_point()

    def to_nvector(self):
        """
        Converts Cartesian position vector to n-vector.

        Returns
        -------
        n_EB_E:  Nvector object
            n-vector(s) [no unit] of position B, decomposed in E.

        The calculation is excact, taking the ellipsity of the Earth into
        account. It is also non-singular as both n-vector and p-vector are
        non-singular (except for the center of the Earth).

        See also
        --------
        n_EB_E2p_EB_E, Nvector
        """
        frame = self.frame
        p_EB_E = self.pvector
        R_Ee = frame.R_Ee
        n_EB_E, depth = p_EB_E2n_EB_E(p_EB_E, a=frame.a, f=frame.f, R_Ee=R_Ee)
        return Nvector(n_EB_E, z=depth, frame=frame)

    def __add__(self, other):
        if self.frame is not other.frame:
            warnings.warn('Frames are possibly unequal')
        return ECEFvector(self.pvector + other.pvector, self.frame)

    def __sub__(self, other):
        if self.frame is not other.frame:
            warnings.warn('Frames are possibly unequal')
        return ECEFvector(self.pvector - other.pvector, self.frame)

    def __neg__(self):
        return ECEFvector(-self.pvector, self.frame)


class GeoPath(object):
    """

    """
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def _euclidean_cross_track_distance(self, cos_angle, radius=1):
        return -cos_angle * radius

    def _great_circle_cross_track_distance(self, cos_angle, radius=1):
        return (arccos(cos_angle) - pi / 2) * radius

    def nvectors(self):
        return self.point1.to_nvector(), self.point2.to_nvector()

    def _nvectors(self):
        n_EA_E, n_EB_E = self.nvectors()
        return n_EA_E.normal, n_EB_E.normal

    def _normal_to_great_circle(self):
        n_EA1_E, n_EA2_E = self._nvectors()
        return cross(n_EA1_E, n_EA2_E, axis=0)

    def _get_average_radius(self):
        p_E1_E = self.point1.to_ecef_vector()
        p_E2_E = self.point2.to_ecef_vector()
        radius = (norm(p_E1_E.pvector, axis=0) +
                  norm(p_E2_E.pvector, axis=0)) / 2
        return radius

    def cross_track_distance(self, point, method='greatcircle', radius=None):
        """
        Return cross track distance from the path to the geo-point.

        Parameters
        ----------
        point: GeoPoint object
            containing geodetic latitude and longitude [rad]
            and depth, z [m], relative to the ellipsoid (depth = -height).
        radius: real scalar
            radius of sphere in [m]. Default mean Earth radius
        method: string
            defining distance calculated. Options are:
            'greatcircle' or 'euclidian'

        Returns
        -------
        distance: real scalar
            distance in [m]
        """
        if radius is None:
            radius = self._get_average_radius()
        c_E = unit(self._normal_to_great_circle())
        n_EB_E = point.to_nvector()
        cos_angle = dot(c_E.T, n_EB_E.normal)
        if method[0] == 'e':
            return self._euclidean_cross_track_distance(cos_angle, radius)
        return self._great_circle_cross_track_distance(cos_angle, radius)

    def track_distance(self, method='greatcircle', radius=None):
        """
        Return the distance of the path.
        """
        if radius is None:
            radius = self._get_average_radius()
        n_EA_E, n_EB_E = self._nvectors()

        if method[0] == "e":  # Euclidean distance:
            return norm(n_EB_E - n_EA_E, axis=0) * radius
        return great_circle_distance(n_EA_E, n_EB_E, radius)

    def intersection(self, path):
        """
        Return the intersection between the paths

        Parameters
        ----------
        path: GeoPath object
            path to intersect

        Returns
        -------
        point: GeoPoint
            point of intersection between paths
        """
        frame = self.point1.frame
        n_EA1_E, n_EA2_E = self._nvectors()
        n_EB1_E, n_EB2_E = path._nvectors()

        # Find the intersection between the two paths, n_EC_E:
        n_EC_E_tmp = unit(cross(cross(n_EA1_E, n_EA2_E, axis=0),
                                cross(n_EB1_E, n_EB2_E, axis=0), axis=0),
                          norm_zero_vector=np.nan)

        # n_EC_E_tmp is one of two solutions, the other is -n_EC_E_tmp. Select
        # the one that is closet to n_EA1_E, by selecting sign from the dot
        # product between n_EC_E_tmp and n_EA1_E:
        n_EC_E = np.sign(dot(n_EC_E_tmp.T, n_EA1_E)) * n_EC_E_tmp
        if np.any(np.isnan(n_EC_E)):
            warnings.warn('Paths are parallell. No intersection point. '
                          'NaN returned.')

        lat_EC, long_EC = n_E2lat_lon(n_EC_E, frame.R_Ee)
        return GeoPoint(lat_EC, long_EC, frame=frame)


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


def get_north_pole_axis_for_E_frame(axis='z'):
    """
    Selects axes of the coordinate frame E.

    R_Ee controls the axes of the coordinate frame E (Earth-Centred,
    Earth-Fixed, ECEF) used by the other functions in this library

    There are two choices of E-axes that are described in Table 2 in Gade
    (2010):

    * e: z-axis points to the North Pole and
         x-axis points to the point where latitude = longitude = 0.
         This choice is very common in many fields.

    * E: x-axis points to the North Pole,
         y-axis points towards longitude +90deg (east) and latitude = 0.
         This choice of axis directions ensures that at zero latitude and
         longitude, N (North-East-Down) has the same orientation as E.
         If roll/pitch/yaw are zero, also B (Body, forward, starboard, down)
         has this orientation. In this manner, the axes of E is chosen to
         correspond with the axes of N and B.

    Based on this we get:
    R_Ee=[0 0 1
          0 1 0
         -1 0 0]

    The above R_Ee should be returned from this function when using z-axis to
    the North pole (which is most common). When using x-axis to the North
    pole, R_Ee should be set to I (identity matrix) (since the functions in
    this library are originally written for this option).

    Reference
    ---------
    Gade, K. (2010). A Nonsingular Horizontal Position Representation,
    The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
    www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf
    """
    R_Ee = NORTH_POLE[axis]
    return R_Ee


def set_north_pole_axis_for_E_frame(axis='z'):
    __doc__ = get_north_pole_axis_for_E_frame.__doc__  # @ReservedAssignment
    global R_Ee
    R_Ee = get_north_pole_axis_for_E_frame(axis)
    return R_Ee


def nthroot(x, n):
    """
    Return the n'th root of x to machine precision
    """
    y = x**(1./n)
    return np.where((x != 0) & (_EPS * np.abs(x) < 1),
                    y - (y**n-x)/(n*y**(n-1)), y)


def deg(rad_angle):
    """
    Converts angle in radians to degrees.

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
    rad
    """
    return rad2deg(rad_angle)


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
    deg
    """
    return deg2rad(deg_angle)


def unit(vector, norm_zero_vector=1):
    """
    Return input vector of unit length, i.e. norm==1.

    Parameters
    ----------
    vector : 3 x m array
        m column vectors

    Returns
    -------
    unitvector : 3 x m array
        normalized unitvector(s) along axis==0.
    """
    current_norm = norm(vector, axis=0)
    unit_vector = vector / current_norm
    idx = np.flatnonzero(current_norm == 0)

    unit_vector[:, idx] = 0 * norm_zero_vector
    unit_vector[0, idx] = 1 * norm_zero_vector
    return unit_vector


def lat_lon2n_E(latitude, longitude, R_Ee=None):
    """
    Converts latitude and longitude to n-vector.

    Parameters
    ----------
    latitude, longitude: real scalars or vectors of length n.
        Geodetic latitude and longitude given in [rad]

    Returns
    -------
    n_E: 3 x n array
        n-vector(s) [no unit] decomposed in E.

    See also
    --------
    n_E2lat_lon.
    """
    if R_Ee is None:
        R_Ee = get_north_pole_axis_for_E_frame()
    # Equation (3) from Gade (2010):
    nvec = np.vstack((sin(latitude),
                      sin(longitude) * cos(latitude),
                      -cos(longitude) * cos(latitude)))
    n_E = np.dot(R_Ee.T, nvec)
    return n_E


def n_E2lat_lon(n_E, R_Ee=None):
    """
    Converts n-vector to latitude and longitude.

    Parameters
    ----------
    n_E: 3 x n array
        n-vector [no unit] decomposed in E.

    Returns
    -------
    latitude, longitude: real scalars or vectors of lengt n.
        Geodetic latitude and longitude given in [rad]

    See also
    --------
    lat_lon2n_E.
    """
    if R_Ee is None:
        R_Ee = get_north_pole_axis_for_E_frame()
    _check_length_deviation(n_E)
    n_E = np.dot(R_Ee, n_E)

    # Equation (5) in Gade (2010):
    longitude = arctan2(n_E[1, :], -n_E[2, :])

    # Equation (6) in Gade (2010) (Robust numerical solution)
    equatorial_component = sqrt(n_E[1, :]**2 + n_E[2, :]**2)
    # vector component in the equatorial plane
    latitude = arctan2(n_E[0, :], equatorial_component)
    # atan() could also be used since latitude is within [-pi/2,pi/2]

    # latitude=asin(n_E[0] is a theoretical solution, but close to the Poles
    # it is ill-conditioned which may lead to numerical inaccuracies (and it
    # will give imaginary results for norm(n_E)>1)
    return latitude, longitude


def _check_length_deviation(n_E, limit=0.1):
    """
    n-vector should have length=1,  i.e. norm(n_E)=1.

    A deviation from 1 exceeding this limit gives a warning.
    This function only depends of the direction of n-vector, thus the warning
    is included only to give a notice in cases where a wrong input is given
    unintentionally (i.e. the input is not even approximately a unit vector).

    If a matrix of n-vectors is input, only first is controlled to save time
    (assuming advanced users input correct n-vectors)
    """
    length_deviation = np.abs(norm(n_E[:, 0]) - 1)
    if length_deviation > limit:
        warnings.warn('n-vector should have unit length: '
                      'norm(n_E)~=1 ! Error is: {}'.format(length_deviation))


def n_E2R_EN(n_E, R_Ee=None):
    """
    Returns the rotation matrix R_EN from n-vector.

    Parameters
    ----------
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E

    Returns
    -------
    R_EN:  3 x 3 array
        The resulting rotation matrix [no unit] (direction cosine matrix).

    See also
    --------
    R_EN2n_E, n_E_and_wa2R_EL, R_EL2n_E.
    """
    if R_Ee is None:
        R_Ee = get_north_pole_axis_for_E_frame()
    _check_length_deviation(n_E)
    n_E = unit(np.dot(R_Ee, n_E))

    # N coordinate frame (North-East-Down) is defined in Table 2 in Gade (2010)
    # Find z-axis of N (Nz):
    Nz_E = -n_E  # z-axis of N (down) points opposite to n-vector

    # Find y-axis of N (East)(remember that N is singular at Poles)
    # Equation (9) in Gade (2010):
    # Ny points perpendicular to the plane
    Ny_E_direction = cross([[1], [0], [0]], n_E, axis=0)
    # formed by n-vector and Earth's spin axis
    outside_poles = (norm(Ny_E_direction) != 0)
    if outside_poles:
        Ny_E = unit(Ny_E_direction)
    else:  # Pole position:
        Ny_E = array([[0], [1], [0]])  # selected y-axis direction

    # Find x-axis of N (North):
    Nx_E = cross(Ny_E, Nz_E, axis=0)  # Final axis found by right hand rule

    # Form R_EN from the unit vectors:
    R_EN = dot(R_Ee.T, np.hstack((Nx_E, Ny_E, Nz_E)))

    return R_EN


def n_E_and_wa2R_EL(n_E, wander_azimuth, R_Ee=None):
    """
    Returns rotation matrix R_EL from n-vector and wander azimuth angle.

    R_EL = n_E_and_wa2R_EL(n_E,wander_azimuth) Calculates the rotation matrix
    (direction cosine matrix) R_EL using n-vector (n_E) and the wander
    azimuth angle.
    When wander_azimuth=0, we have that N=L (See Table 2 in Gade (2010) for
    details)

    Parameters
    ----------
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E
    wander_azimuth: real scalar
        Angle [rad] between L's x-axis and north, positive about L's z-axis.

    Returns
    -------
    R_EL: 3 x 3 array
        The resulting rotation matrix.       [no unit]

    See also
    --------
    R_EL2n_E, R_EN2n_E, n_E2R_EN.
    """
    if R_Ee is None:
        R_Ee = get_north_pole_axis_for_E_frame()
    latitude, longitude = n_E2lat_lon(n_E, R_Ee)

    # Reference: See start of Section 5.2 in Gade (2010):
    R_EL = dot(R_Ee.T, xyz2R(longitude, -latitude, wander_azimuth))
    return R_EL


def _check_backward_compatibility(a, f):
    """
    Previously, custom ellipsoid was spesified by a and b.
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


def n_EB_E2p_EB_E(n_EB_E, depth=0, a=6378137, f=1.0/298.257223563, R_Ee=None):
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

    Returns
    -------
    p_EB_E:  3 x n array
        Cartesian position vector(s) from E to B, decomposed in E.

    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, n_EB_E. The function converts
    to cartesian position vector ("ECEF-vector"), p_EB_E, in meters.
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    See also
    --------
    p_EB_E2n_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
    """
    if R_Ee is None:
        R_Ee = get_north_pole_axis_for_E_frame()
    _check_length_deviation(n_EB_E)

    n_EB_E = unit(dot(R_Ee, n_EB_E))
    if depth is None:
        depth = np.zeros((1, np.shape(n_EB_E)[1]))

    a, f = _check_backward_compatibility(a, f)
    b = a * (1 - f)  # semi-minor axis

    # The following code implements equation (22) in Gade (2010):
    scale = np.vstack((1,
                       (1 - f),
                       (1 - f)))
    denominator = norm(n_EB_E / scale, axis=0)

    # We first calculate the position at the origin of coordinate system L,
    # which has the same n-vector as B (n_EL_E = n_EB_E),
    # but lies at the surface of the Earth (z_EL = 0).

    p_EL_E = b / denominator * n_EB_E / scale**2
    p_EB_E = dot(R_Ee.T, p_EL_E - n_EB_E * depth)

    return p_EB_E


def p_EB_E2n_EB_E(p_EB_E, a=6378137, f=1.0/298.257223563, R_Ee=None):
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

    Returns
    -------
    n_EB_E:  3 x n array
        n-vector(s) [no unit] of position B, decomposed in E.
    depth:  1 x n array
        Depth(s) [m] of system B, relative to the ellipsoid (depth = -height)

    The position of B (typically body) relative to E (typically Earth) is
    given into this function as cartesian position vector p_EB_E, in meters.
    ("ECEF-vector"). The function converts to n-vector, n_EB_E and its
    depth, depth.
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    See also
    --------
    n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
    """
    if R_Ee is None:
        R_Ee = get_north_pole_axis_for_E_frame()
    p_EB_E = dot(R_Ee, p_EB_E)
    # R_Ee selects correct E-axes, see R_Ee.m for details
    a, f = _check_backward_compatibility(a, f)

    # e_2 = eccentricity**2
    e_2 = 2 * f - f**2  # = 1-b**2/a**2

    # The following code implements equation (23) from Gade (2010):
    R_2 = p_EB_E[1, :]**2 + p_EB_E[2, :]**2
    R = sqrt(R_2)   # R = component of p_EB_E in the equatorial plane

    p = R_2 / a**2
    q = (1 - e_2) / (a**2) * p_EB_E[0, :]**2
    r = (p + q - e_2**2) / 6

    s = e_2**2 * p * q / (4 * r**3)
    t = nthroot((1 + s + sqrt(s*(2+s))), 3)
    # t = (1 + s + sqrt(s * (2 + s)))**(1. / 3)
    u = r * (1 + t + 1. / t)
    v = sqrt(u**2 + e_2**2 * q)

    w = e_2 * (u + v - q) / (2 * v)
    k = sqrt(u + v + w**2) - w
    d = k * R / (k + e_2)

    # Calculate height:
    height = (k + e_2 - 1) / k * sqrt(d**2 + p_EB_E[0, :]**2)

    temp = 1. / sqrt(d**2 + p_EB_E[0, :]**2)

    n_EB_E_x = temp * p_EB_E[0, :]
    n_EB_E_y = temp * k / (k + e_2) * p_EB_E[1, :]
    n_EB_E_z = temp * k / (k + e_2) * p_EB_E[2, :]

    n_EB_E = np.vstack((n_EB_E_x, n_EB_E_y, n_EB_E_z))
    n_EB_E = unit(dot(R_Ee.T, n_EB_E))  # Ensure unit length:

    depth = -height
    return n_EB_E, depth


def n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=6378137,
                             f=1.0/298.257223563):
    """
    From two positions A and B, finds the delta position.

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

    Returns
    -------
    p_AB_E:  3 x n array
        Cartesian position vector(s) from A to B, decomposed in E.

    The n-vectors for positions A (n_EA_E) and B (n_EB_E) are given. The
    output is the delta vector from A to B (p_AB_E).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E.
    """

    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f)
    p_EB_E = n_EB_E2p_EB_E(n_EB_E, z_EB, a, f)
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


def n_EA_E_and_p_AB_E2n_EB_E(n_EA_E, p_AB_E, z_EA=0, a=6378137,
                             f=1.0/298.257223563, R_Ee=None):
    """
    From position A and delta, finds position B.

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

    Returns
    -------
    n_EB_E:  3 x n array
        n-vector(s) [no unit] of position B, decomposed in E.
    z_EB:  1 x n array
        Depth(s) [m] of system B, relative to the ellipsoid.
        (z_EB = -height)

    The n-vector for position A (n_EA_E) and the position-vector from position
    A to position B (p_AB_E) are given. The output is the n-vector of position
    B (n_EB_E) and depth of B (z_EB).
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).
    The default ellipsoid model used is WGS-84, but other ellipsoids/spheres
    might be specified.

    See also
    --------
    n_EA_E_and_n_EB_E2p_AB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E.
    """
    if R_Ee is None:
        R_Ee = get_north_pole_axis_for_E_frame()
    a, f = _check_backward_compatibility(a, f)

    # Function 2. in Section 5.4 in Gade (2010):
    p_EA_E = n_EB_E2p_EB_E(n_EA_E, z_EA, a, f, R_Ee)
    p_EB_E = p_EA_E + p_AB_E
    n_EB_E, z_EB = p_EB_E2n_EB_E(p_EB_E, a, f, R_Ee)
    return n_EB_E, z_EB


def R2xyz(R_AB):
    """
    Returns the angles about new axes in the xyz-order from a rotation matrix.

    Parameters
    ----------
    R_AB: 3x3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

    Returns
    -------
    x, y, z: real scalars
        Angles [rad] of rotation about new axes.

    The x, y, z angles are called Euler angles or Tait-Bryan angles and are
    defined by the following procedure of successive rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle x about its x-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle z about its NEWEST z-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    See also
    --------
    xyz2R, R2zyx, xyz2R.
    """
    z = arctan2(-R_AB[0, 1], R_AB[0, 0])  # atan2: [-pi pi]
    x = arctan2(-R_AB[1, 2], R_AB[2, 2])

    sin_y = R_AB[0, 2]

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = sqrt((R_AB[0, 0]**2 + R_AB[0, 1]**2 +
                  R_AB[1, 2]**2 + R_AB[2, 2]**2)/2)

    y = arctan2(sin_y, cos_y)
    return x, y, z


def R2zyx(R_AB):
    """
    Returns the angles about new axes in the zxy-order from a rotation matrix.

    Parameters
    ----------
    R_AB:  3x3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

    Returns
    -------
    z, y, x: real scalars
        Angles [rad] of rotation about new axes.

    The z, x, y angles are called Euler angles or Tait-Bryan angles and are
    defined by the following procedure of successive rotations:
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

    See also
    --------
    zyx2R, xyz2R, R2xyz.
    """

    z = arctan2(R_AB[1, 0], R_AB[0, 0])  # atan2: [-pi pi]
    x = arctan2(R_AB[2, 1], R_AB[2, 2])

    sin_y = -R_AB[2, 0]

    # cos_y is based on as many elements as possible, to average out
    # numerical errors. It is selected as the positive square root since
    # y: [-pi/2 pi/2]
    cos_y = sqrt((R_AB[0, 0]**2 + R_AB[1, 0]**2 +
                  R_AB[2, 1]**2 + R_AB[2, 2]**2)/2)

    y = arctan2(sin_y, cos_y)
    return z, y, x


def R_EL2n_E(R_EL):
    """
    Returns n-vector from the rotation matrix R_EL.

    Parameters
    ----------
    R_EL: 3 x 3 array
        Rotation matrix (direction cosine matrix) [no unit]

    Returns
    -------
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E.

    See also R_EN2n_E, n_E_and_wa2R_EL, n_E2R_EN.
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = dot(R_EL, np.vstack((0, 0, -1)))
    return n_E


def R_EN2n_E(R_EN):
    """
    Returns n-vector from the rotation matrix R_EN.

    Parameters
    ----------
    R_EN: 3 x 3 array
        Rotation matrix (direction cosine matrix) [no unit]

    Returns
    -------
    n_E: 3 x 1 array
        n-vector [no unit] decomposed in E.

    See also
    --------
    n_E2R_EN, R_EL2n_E, n_E_and_wa2R_EL.
    """
    # n-vector equals minus the last column of R_EL and R_EN, see Section 5.5
    # in Gade (2010)
    n_E = dot(R_EN, np.vstack((0, 0, -1)))
    return n_E


def xyz2R(x, y, z):
    """
    Returns rotation matrix from 3 angles about new axes in the xyz-order.

    Parameters
    ----------
    x,y,z: real scalars
        Angles [rad] of rotation about new axes.

    Returns
    -------
    R_AB: 3 x 3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

    The rotation matrix R_AB is created based on 3 angles x,y,z about new axes
    (intrinsic) in the order x-y-z. The angles are called Euler angles or
    Tait-Bryan angles and are defined by the following procedure of successive
    rotations:
    Given two arbitrary coordinate frames A and B. Consider a temporary frame
    T that initially coincides with A. In order to make T align with B, we
    first rotate T an angle x about its x-axis (common axis for both A and T).
    Secondly, T is rotated an angle y about the NEW y-axis of T. Finally, T
    is rotated an angle z about its NEWEST z-axis. The final orientation of
    T now coincides with the orientation of B.

    The signs of the angles are given by the directions of the axes and the
    right hand rule.

    See also
    --------
    R2xyz, zyx2R, R2zyx.
    """
    cz, sz = cos(z), sin(z)
    cy, sy = cos(y), sin(y)
    cx, sx = cos(x), sin(x)

    R_AB = array([[cy * cz, -cy * sz, sy],
                  [sy*sx*cz + cx*sz, -sy*sx*sz + cx*cz, -cy*sx],
                  [-sy*cx*cz + sx*sz, sy*cx*sz + sx*cz, cy*cx]])

    return np.squeeze(R_AB)


def zyx2R(z, y, x):
    """
    Returns rotation matrix from 3 angles about new axes in the zyx-order.

    Parameters
    ----------
    z, y, x: real scalars
        Angles [rad] of rotation about new axes.

    Returns
    -------
    R_AB: 3 x 3 array
        rotation matrix [no unit] (direction cosine matrix) such that the
        relation between a vector v decomposed in A and B is given by:
        v_A = np.dot(R_AB, v_B)

    The rotation matrix R_AB is created based on 3 angles
    z,y,x about new axes (intrinsic) in the order z-y-x. The angles are called
    Euler angles or Tait-Bryan angles and are defined by the following
    procedure of successive rotations:
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

    See also
    --------
    R2zyx, xyz2R, R2xyz.
    """
    cz, sz = cos(z), sin(z)
    cy, sy = cos(y), sin(y)
    cx, sx = cos(x), sin(x)

    R_AB = array([[cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy*cx],
                  [sz * cy,  cz * cx + sz * sy * sx, - cz * sx + sz*sy*cx],
                  [-sy, cy * sx, cy * cx]])

    return np.squeeze(R_AB)


def great_circle_distance(n_EA_E, n_EB_E, radius=6371009.0):
    return arctan2(norm(cross(n_EA_E, n_EB_E, axis=0), axis=0),
                   dot(n_EA_E.T, n_EB_E)) * radius


def azimuth(n_EA_E, n_EB_E, a=6378137, f=1.0/298.257223563):
    """
    Return direction (azimuth) from A to B, relative to North:

    Parameters
    ----------
    n_EA_E, n_EB_E:  3 x n array
        n-vector(s) [no unit] of position A and B, respectively,
        decomposed in E.
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.

    Returns
    -------
    azimuth: n, array
        Angle the line makes with a meridian, taken clockwise from north.
    """
    # Step2: Find p_AB_E (delta decomposed in E).
    p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA=0, z_EB=0, a=a, f=f)

    # Step3: Find R_EN for position A:
    R_EN = n_E2R_EN(n_EA_E)

    # Step4: Find p_AB_N
    p_AB_N = dot(R_EN.T, p_AB_E)
    # (Note the transpose of R_EN: The "closest-rule" says that when
    # decomposing, the frame in the subscript of the rotation matrix that
    # is closest to the vector, should equal the frame where the vector is
    # decomposed. Thus the calculation np.dot(R_NE, p_AB_E) is correct,
    # since the vector is decomposed in E, and E is closest to the vector.
    # In the example we only had R_EN, and thus we must transpose it:
    # R_EN'=R_NE)

    # Step5: Also find the direction (azimuth) to B, relative to north:
    return arctan2(p_AB_N[1], p_AB_N[0])


def distance_rad_bearing_rad2point(n_EA_E, distance_rad, bearing_rad):
    k_east_E = unit(cross(dot(R_Ee.T, [[1], [0], [0]]), n_EA_E, axis=0))
    k_north_E = cross(n_EA_E, k_east_E, axis=0)

    # Step2: Find the initial direction vector d_E:
    d_E = k_north_E * cos(bearing_rad) + k_east_E * sin(bearing_rad)

    # Step3: Find n_EB_E:
    n_EB_E = n_EA_E * cos(distance_rad) + d_E * sin(distance_rad)
    return n_EB_E


def horizontal_mean_position(n_EB_E):
    """
    Return the n-vector of the horizontal mean position.

    Parameters
    ----------
    n_EB_E:  3 x n array
        n-vectors [no unit] of positions Bi, decomposed in E.

    Returns
    -------
    p_EM_E:  3 x 1 array
        n-vector [no unit] of the mean positions of all Bi, decomposed in E.
    """
    n_EM_E = unit(np.sum(n_EB_E, axis=1).reshape((3, 1)))
    return n_EM_E


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    print('Docstrings tested')


if __name__ == "__main__":
    test_docstrings()
    # print('{:15.15f}'.format(nthroot(27., 3.)-0.0))
