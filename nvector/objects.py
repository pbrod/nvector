'''
Created on 29. des. 2015

@author: pab
'''
from __future__ import division, print_function
import numpy as np
from numpy import pi, arccos, cross, dot
from numpy.linalg import norm
from geographiclib.geodesic import Geodesic as _Geodesic
from nvector._core import (select_ellipsoid, NORTH_POLE, rad, deg, zyx2R,
                           lat_lon2n_E, n_E2lat_lon, n_E2R_EN, n_E_and_wa2R_EL,
                           n_EB_E2p_EB_E, p_EB_E2n_EB_E, unit,
                           great_circle_distance)
import warnings

__all__ = ['FrameE', 'FrameB', 'FrameL', 'FrameN', 'GeoPoint', 'GeoPath',
           'Nvector', 'Pvector', 'ECEFvector', 'diff_positions']


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

    def GeoPoint(self, *args, **kwds):
        kwds.pop('frame', None)
        return GeoPoint(*args, frame=self, **kwds)

    def Nvector(self, *args, **kwds):
        kwds.pop('frame', None)
        return Nvector(*args, frame=self, **kwds)

    def ECEFvector(self, *args, **kwds):
        kwds.pop('frame', None)
        return ECEFvector(*args, frame=self, **kwds)


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
    def __init__(self, position):
        nvector = position.to_nvector()
        n_EA_E = nvector.normal
        self.nvector = Nvector(n_EA_E, z=0, frame=nvector.frame)
        self.R_EN = n_E2R_EN(n_EA_E, nvector.frame.R_Ee)

    def _is_equal_to(self, other):
        return (np.allclose(self.R_EN, other.R_EN) and
                self.nvector == other.nvector)

    def Pvector(self, pvector):
        return Pvector(pvector, frame=self)


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
    def __init__(self, position, wander_azimuth=0):
        nvector = position.to_nvector()
        n_EA_E = nvector.normal
        R_Ee = nvector.frame.R_Ee
        self.nvector = Nvector(n_EA_E, z=0, frame=nvector.frame)
        self.R_EN = n_E_and_wa2R_EL(n_EA_E, wander_azimuth, R_Ee=R_Ee)


class FrameB(FrameN):
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
    def __init__(self, position, yaw=0, pitch=0, roll=0, degrees=False):
        nvector = position.to_nvector()
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


def frame_definitions():
    """
    Coordinate frame definitions
    ----------------------------

    """ + FrameE.__doc__ + FrameB.__doc__ + FrameN.__doc__ + FrameL.__doc__
    pass


def _check_frames(self, other):
    if not self.frame == other.frame:
        raise ValueError('Frames are unequal')


def _default_frame(frame):
    if frame is None:
        return FrameE()
    return frame


class GeoPoint(object):
    """
    Geographical position given as latitude, longitude, depth in frame E

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
    >>> wgs84 = nv.FrameE(name='WGS84')

    # The geodesic inverse problem
    >>> point1 = wgs84.GeoPoint(-41.32, 174.81, degrees=True))
    >>> point2 = wgs84.GeoPoint(40.96, -5.50, degrees=True)
    >>> s12, az1, az2 = point1.distance_and_azimuth(point2, degrees=True)
    >>> 's12 = {:5.2f}, az1 = {:5.2f}, az2 = {:5.2f}'.format(s12, az1, az2)
    's12 = 19959679.27, az1 = 161.07, az2 = 18.83'

    # The geodesic direct problem
    >>> point1 = wgs84.GeoPoint(40.6, -73.8, degrees=True)
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
        self.frame = _default_frame(frame)

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
        _check_frames(self, point)

        lat_a, lon_a = self.latitude, self.longitude
        lat_b, lon_b = point.latitude, point.longitude
        if degrees:
            lat_a, lon_a, lat_b, lon_b = deg((lat_a, lon_a, lat_b, lon_b))
        return self.frame.inverse(lat_a, lon_a, lat_b, lon_b, z=self.z,
                                  long_unroll=long_unroll, degrees=degrees)


class Nvector(object):
    """
    Geographical position given as N-vector and depth in frame E

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
        self.frame = _default_frame(frame)

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

    def to_nvector(self):
        return self

    def mean_horizontal_position(self):
        """
        Return the n-vector of the horizontal mean position.

        Returns
        -------
        p_EM_E:  3 x 1 array
            n-vector [no unit] of the mean position, decomposed in E.
        """
        n_EB_E = self.normal
        n_EM_E = unit(np.sum(n_EB_E, axis=1).reshape((3, 1)))
        return self.frame.Nvector(n_EM_E)

    def __eq__(self, other):
        try:
            if self is other:
                return True
            return self._is_equal_to(other)
        except AttributeError:
            return False
        raise ValueError

    def _is_equal_to(self, other):
        return (np.allclose(self.normal, other.normal) and
                self.frame == other.frame)


def diff_positions(pointA, pointB):
    """
    Return delta position from two positions A and B.

    Parameters
    ----------
    pointA, pointB: Nvector, GeoPoint or ECEFvector objects
        position A and B, decomposed in E.

    Returns
    -------
    p_AB_E:  ECEFvector
        Cartesian position vector(s) from A to B, decomposed in E.

    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).

    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E.
    """
    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = pointA.to_ecef_vector()
    p_EB_E = pointB.to_ecef_vector()
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
    Geographical position given as Cartesian position vector in frame E

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
        self.frame = _default_frame(frame)

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
        _check_frames(self, frame.nvector)
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
        _check_frames(self, other)
        return ECEFvector(self.pvector + other.pvector, self.frame)

    def __sub__(self, other):
        _check_frames(self, other)
        return ECEFvector(self.pvector - other.pvector, self.frame)

    def __neg__(self):
        return ECEFvector(-self.pvector, self.frame)


class GeoPath(object):
    """
    Geographical path between two points in Frame E

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
        Return cross track distance from the path to a point.

        Parameters
        ----------
        point: GeoPoint, Nvector or ECEFvector object
            position to measure the cross track distance to.
        radius: real scalar
            radius of sphere in [m]. Default mean Earth radius
        method: string
            defining distance calculated. Options are:
            'greatcircle' or 'euclidean'

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
        if method[0].lower() == 'e':
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

if __name__ == '__main__':
    pass
