"""
Created on 29. des. 2015

@author: pab
"""
from __future__ import division, print_function
import numpy as np
from numpy import pi, arccos, cross, dot
from numpy.linalg import norm
from geographiclib.geodesic import Geodesic as _Geodesic
from nvector._core import (select_ellipsoid, rad, deg, zyx2R,
                           lat_lon2n_E, n_E2lat_lon, n_E2R_EN, n_E_and_wa2R_EL,
                           n_EB_E2p_EB_E, p_EB_E2n_EB_E, unit,
                           great_circle_distance, mean_horizontal_position,
                           E_rotation)
import warnings

__all__ = ['FrameE', 'FrameB', 'FrameL', 'FrameN', 'GeoPoint', 'GeoPath',
           'Nvector', 'ECEFvector', 'Pvector', 'diff_positions']


_GEO_POINT_DOC = """
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

    The geodesic inverse problem

        >>> point1 = wgs84.GeoPoint(-41.32, 174.81, degrees=True))
        >>> point2 = wgs84.GeoPoint(40.96, -5.50, degrees=True)
        >>> s12, az1, az2 = point1.distance_and_azimuth(point2, degrees=True)
        >>> 's12 = {:5.2f}, az1 = {:5.2f}, az2 = {:5.2f}'.format(s12, az1, az2)
        's12 = 19959679.27, az1 = 161.07, az2 = 18.83'

    The geodesic direct problem

        >>> point1 = wgs84.GeoPoint(40.6, -73.8, degrees=True)
        >>> az1, distance = 45, 10000e3
        >>> point2, az2 = point1.geo_point(distance, az1, degrees=True)
        >>> lat2, lon2 = point2.latitude_deg, point2.longitude_deg
        >>> msg = 'lat2 = {:5.2f}, lon2 = {:5.2f}, az2 = {:5.2f}'
        >>> msg.format(lat2, lon2, az2)
        'lat2 = 32.64, lon2 = 49.01, az2 = 140.37'

    """

_NVECTOR_DOC = """
    Geographical position given as n-vector and depth in frame E

    Parameters
    ----------
    normal: 3 x n array
        n-vector(s) [no unit] decomposed in E.
    z: real scalar or vector of length n.
        Depth(s) [m]  relative to the ellipsoid (depth = -height)
    frame: FrameE object
        reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, n_EB_E and a depth, z relative to the
    ellipsiod.

    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """

_ECEFVECTOR_DOC = """
    Geographical position given as Cartesian position vector in frame E

    Parameters
    ----------
    pvector: 3 x n array
        Cartesian position vector(s) [m] from E to B, decomposed in E.
    frame: FrameE object
        reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as p-vector, p_EB_E relative to the center of the
    frame.

    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """


class _BaseFrame(object):
    def __eq__(self, other):
        try:
            if self is other:
                return True
            return self._is_equal_to(other)
        except AttributeError:
            return False


class FrameE(_BaseFrame):

    """
    Earth-fixed frame

    Parameters
    ----------
    a: real scalar, default WGS-84 ellipsoid.
        Semi-major axis of the Earth ellipsoid given in [m].
    f: real scalar, default WGS-84 ellipsoid.
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a is used in stead of WGS-84.
    name: string
        defining the default ellipsoid.
    axes: 'e' or 'E'
        defines axes orientation of E frame. Default is axes='e' which means
        that the orientation of the axis is such that:
        z-axis -> North Pole, x-axis -> Latitude=Longitude=0.

    Notes
    -----
    The frame is Earth-fixed (rotates and moves with the Earth) where the
    origin coincides with Earth's centre (geometrical centre of ellipsoid
    model).

    See also
    --------
    FrameN, FrameL, FrameB
    """
    def __init__(self, a=None, f=None, name='WGS84', axes='e'):
        if a is None or f is None:
            a, f, _full_name = select_ellipsoid(name)
        self.a = a
        self.f = f
        self.name = name
        self.R_Ee = E_rotation(axes)

    def _is_equal_to(self, other):
        return (np.allclose(self.a, other.a) and
                np.allclose(self.f, other.f) and
                np.allclose(self.R_Ee, other.R_Ee))

    def inverse(self, lat_a, lon_a, lat_b, lon_b, z=0, long_unroll=False,
                degrees=False):
        """
        Return ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        lat_a, lon_a:  real scalars
            Latitude and longitude of position a.
        lat_b, lon_b:  real scalars
            Latitude and longitude of position b.
        z : real scalar
            depth relative to Earth ellipsoid.
        degrees: bool
            angles are given in degrees if True otherwise in radians.

        Returns
        -------
        s_ab: real scalar
            ellipsoidal distance [m] between position A and B.
        azimuth_a, azimuth_b
            direction [rad or deg] of line at position A and B relative to
            North, respectively.

        References
        ----------
        C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_

        """

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
        """
        Return position B computed from position A, distance and azimuth.

        Parameters
        ----------
        lat_a, lon_a:  real scalars
            Latitude and longitude [rad or deg] of position a.
        azimuth_a:
            azimuth [rad or deg] of line at position A.
        distance: real scalar
            ellipsoidal distance [m] between position A and B.
        z : real scalar
            depth relative to Earth ellipsoid.
        degrees: bool
            angles are given in degrees if True otherwise in radians.

        Returns
        -------
        lat_b, lon_b:  real scalars
            Latitude and longitude of position b.
        azimuth_b
            azimuth [rad or deg] of line at position B.

        References
        ----------
        C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_
        """

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
        __doc__ = _GEO_POINT_DOC  # @ReservedAssignment
        kwds.pop('frame', None)
        return GeoPoint(*args, frame=self, **kwds)

    def Nvector(self, *args, **kwds):
        __doc__ = _NVECTOR_DOC  # @ReservedAssignment
        kwds.pop('frame', None)
        return Nvector(*args, frame=self, **kwds)

    def ECEFvector(self, *args, **kwds):
        __doc__ = _ECEFVECTOR_DOC  # @ReservedAssignment
        kwds.pop('frame', None)
        return ECEFvector(*args, frame=self, **kwds)


class FrameN(_BaseFrame):

    """
    North-East-Down frame

    Parameters
    ----------
    position: ECEFvector, GeoPoint or Nvector object
        position of the vehicle (B) which also defines the origin of the local
        frame N. The origin is directly beneath or above the vehicle (B), at
        Earth's surface (surface of ellipsoid model).

    Notes
    -----
    The Cartesian frame is local and oriented North-East-Down, i.e.,
    the x-axis points towards north, the y-axis points towards east (both are
    horizontal), and the z-axis is pointing down.

    When moving relative to the Earth, the frame rotates about its z-axis
    to allow the x-axis to always point towards north. When getting close
    to the poles this rotation rate will increase, being infinite at the
    poles. The poles are thus singularities and the direction of the
    x- and y-axes are not defined here. Hence, this coordinate frame is
    NOT SUITABLE for general calculations.

    Examples
    --------

    **Example 1: "A and B to delta"**
    .. image:: http://www.navlab.net/images/ex1img.png

    Given two positions, A and B as latitudes, longitudes and depths relative
    to Earth, E.

    Find the exact vector between the two positions, given in meters north,
    east, and down.

    Solution:
        >>> import numpy as np
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> pointA = wgs84.GeoPoint(latitude=1, longitude=2, z=3, degrees=True)
        >>> pointB = wgs84.GeoPoint(latitude=4, longitude=5, z=6, degrees=True)

    Step 1: Find p_AB_E (delta decomposed in E).
        >>> p_AB_E = nv.diff_positions(pointA, pointB)

    Step 2: Find p_AB_N (delta decomposed in N).
        >>> frame_N = nv.FrameN(pointA)
        >>> p_AB_N = p_AB_E.change_frame(frame_N)
        >>> p_AB_N = p_AB_N.pvector.ravel()
        >>> valtxt = '{0:8.2f}, {1:8.2f}, {2:8.2f}'.format(*p_AB_N)
        >>> 'Ex1: delta north, east, down = {}'.format(valtxt)
        'Ex1: delta north, east, down = 331730.23, 332997.87, 17404.27'

    See also `Example 1 at www.navlab.net
        <http://www.navlab.net/nvector/#example_1>`_

    See also
    --------
    FrameE, FrameL, FrameB
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
    Local level, Wander azimuth frame

    Parameters
    ----------
    position: ECEFvector, GeoPoint or Nvector object
        position of the vehicle (B) which also defines the origin of the local
        frame L. The origin is directly beneath or above the vehicle (B), at
        Earth's surface (surface of ellipsoid model).
    wander_azimuth: real scalar
        Angle between the x-axis of L and the north direction.

    Notes
    -----
    The Cartesian frame is local and oriented Wander-azimuth-Down. This means
    that the z-axis is pointing down. Initially, the x-axis points towards
    north, and the y-axis points towards east, but as the vehicle moves they
    are not rotating about the z-axis (their angular velocity relative to the
    Earth has zero component along the z-axis).

    (Note: Any initial horizontal direction of the x- and y-axes is valid
    for L, but if the initial position is outside the poles, north and east
    are usually chosen for convenience.)

    The L-frame is equal to the N-frame except for the rotation about the
    z-axis, which is always zero for this frame (relative to E). Hence, at
    a given time, the only difference between the frames is an angle
    between the x-axis of L and the north direction; this angle is called
    the wander azimuth angle. The L-frame is well suited for general
    calculations, as it is non-singular.

    See also
    --------
    FrameE, FrameN, FrameB
    """
    def __init__(self, position, wander_azimuth=0):
        nvector = position.to_nvector()
        n_EA_E = nvector.normal
        R_Ee = nvector.frame.R_Ee
        self.nvector = Nvector(n_EA_E, z=0, frame=nvector.frame)
        self.R_EN = n_E_and_wa2R_EL(n_EA_E, wander_azimuth, R_Ee=R_Ee)


class FrameB(FrameN):

    """
    Body frame

    Parameters
    ----------
    position: ECEFvector, GeoPoint or Nvector object
        position of the vehicle's reference point which also coincides with
        the origin of the frame B.
    yaw, pitch, roll: real scalars
        defining the orientation of frame B in [deg] or [rad].
    degrees : bool
        if True yaw, pitch, roll are given in degrees otherwise in radians

    Notes
    -----
    The frame is fixed to the vehicle where the x-axis points forward, the
    y-axis to the right (starboard) and the z-axis in the vehicle's down
    direction.

    See also
    --------
    FrameE, FrameL, FrameN
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


def _check_frames(self, other):
    if not self.frame == other.frame:
        raise ValueError('Frames are unequal')


def _default_frame(frame):
    if frame is None:
        return FrameE()
    return frame


class GeoPoint(object):
    __doc__ = _GEO_POINT_DOC

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
        """
        Converts latitude and longitude to ECEF-vector.
        """
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
    __doc__ = _NVECTOR_DOC

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
        Return horizontal mean position of the n-vectors.

        Returns
        -------
        p_EM_E:  3 x 1 array
            n-vector [no unit] of the mean position, decomposed in E.
        """
        n_EB_E = self.normal
        n_EM_E = mean_horizontal_position(n_EB_E)
        return self.frame.Nvector(n_EM_E)

    def __eq__(self, other):
        try:
            if self is other:
                return True
            return self._is_equal_to(other)
        except AttributeError:
            return False

    def _is_equal_to(self, other):
        return (np.allclose(self.normal, other.normal) and
                self.frame == other.frame)

    def __add__(self, other):
        _check_frames(self, other)
        return self.frame.Nvector(self.normal + other.normal,
                                  self.z + other.z)

    def __sub__(self, other):
        _check_frames(self, other)
        return self.frame.Nvector(self.normal - other.normal,
                                  self.z - other.z)

    def __neg__(self):
        return self.frame.Nvector(-self.normal, -self.z)

    def __mul__(self, scalar):
        """elementwise multiplication"""

        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal*scalar, self.z*scalar)
        raise NotImplementedError('Only scalar multiplication is implemented')

    def __div__(self, scalar):
        """elementwise division"""
        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal/scalar, self.z/scalar)
        raise NotImplementedError('Only scalar division is implemented')

    __truediv__ = __div__
    __radd__ = __add__
    __rmul__ = __mul__


def diff_positions(pointA, pointB):
    """
    Return delta vector from positions A to B.

    Parameters
    ----------
    pointA, pointB: Nvector, GeoPoint or ECEFvector objects
        position A and B, decomposed in E.

    Returns
    -------
    p_AB_E:  ECEFvector
        Cartesian position vector(s) from A to B, decomposed in E.

    Notes
    -----
    The calculation is excact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).

    Examples
    --------

    **Example 1 "A and B to delta"**
    .. image:: http://www.navlab.net/images/ex1img.png

    Given two positions, A and B as latitudes, longitudes and depths relative
    to Earth, E.

    Find the exact vector between the two positions, given in meters north,
    east, and down.

    Solution:
        >>> import numpy as np
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> pointA = wgs84.GeoPoint(latitude=1, longitude=2, z=3, degrees=True)
        >>> pointB = wgs84.GeoPoint(latitude=4, longitude=5, z=6, degrees=True)

    Step 1: Find p_AB_E (delta decomposed in E).
        >>> p_AB_E = nv.diff_positions(pointA, pointB)

    Step 2: Find p_AB_N (delta decomposed in N).
        >>> frame_N = nv.FrameN(pointA)
        >>> p_AB_N = p_AB_E.change_frame(frame_N)
        >>> p_AB_N = p_AB_N.pvector.ravel()
        >>> valtxt = '{0:8.2f}, {1:8.2f}, {2:8.2f}'.format(*p_AB_N)
        >>> 'Ex1: delta north, east, down = {}'.format(valtxt)
        'Ex1: delta north, east, down = 331730.23, 332997.87, 17404.27'

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
    __doc__ = _ECEFVECTOR_DOC

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
        Converts ECEF-vector to n-vector.

        Returns
        -------
        n_EB_E:  Nvector object
            n-vector(s) [no unit] of position B, decomposed in E.

        Notes
        -----
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

    Parameters
    ----------
     point1, point2: Nvector, GeoPoint or ECEFvector objects
        The path is defined by the line between position 1 and 2, decomposed
        in E.

    Examples
    --------

    **Example 5 "Surface distance"**

    .. image http://www.navlab.net/images/ex5img.png

    Find the surface distance sAB (i.e. great circle distance) between
    positions A and B. Use Earth radius 6371e3 m.

    Great circle solution:
        >>> import nvector as nv
        >>> frame_E = nv.FrameE(a=6371e3, f=0)
        >>> positionA = frame_E.GeoPoint(88, 0, degrees=True)
        >>> positionB = frame_E.GeoPoint(89, -170, degrees=True)

        >>> path = nv.GeoPath(positionA, positionB)
        >>> s_AB2 = path.track_distance(method='greatcircle').ravel()
        >>> d_AB2 = path.track_distance(method='euclidean').ravel()
        >>> msg.format(s_AB2[0] / 1000, d_AB2[0] / 1000)
        'Ex5: Great circle and Euclidean distance = 332.46 km, 332.42 km'

    See also Example 5 at http://www.navlab.net/nvector/#example_5

    **Example 6 "Interpolated position"**

    .. image http://www.navlab.net/images/ex6img.png

    Given the position of B at time t0 and t1, n_EB_E(t0) and n_EB_E(t1).

    Find an interpolated position at time ti, n_EB_E(ti).

    Solution:
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> positionB0 = wgs84.GeoPoint(89, 0, degrees=True)
        >>> positionB1 = wgs84.GeoPoint(89, 180, degrees=True)
        >>> path = nv.GeoPath(positionB0, positionB1)

        >>> t0 = 10.  # time at position B0
        >>> t1 = 20.  # time at position B1
        >>> ti = 16.  # time of interpolation
        >>> ti_n = (ti - t0) / (t1 - t0) # normalized time of interpolation
        >>> g_EB_E_ti = path.interpolate(ti_n).to_geo_point()
        >>> lat_ti, lon_ti = g_EB_E_ti.latitude_deg, g_EB_E_ti.longitude_deg
        >>> msg = 'Ex6, Interpolated position: lat, long = {} deg, {} deg'
        >>> msg.format(lat_ti, lon_ti)
        'Ex6, Interpolated position: lat, long = [ 89.7999805] deg, [ 180.] deg'

    See also Example 6 at http://www.navlab.net/nvector/#example_6

    **Example 9 "Intersection of two paths"**

    Define a path from two given positions (at the surface of a spherical
    Earth), as the great circle that goes through the two points.

    Find the position C where the two paths A and B intersect.

    Solution:
        >>> import nvector as nv~
        >>> pointA1 = nv.GeoPoint(10, 20, degrees=True)
        >>> pointA2 = nv.GeoPoint(30, 40, degrees=True)
        >>> pointB1 = nv.GeoPoint(50, 60, degrees=True)
        >>> pointB2 = nv.GeoPoint(70, 80, degrees=True)
        >>> pathA = nv.GeoPath(pointA1, pointA2)
        >>> pathB = nv.GeoPath(pointB1, pointB2)
        >>> pointC = pathA.intersection(pathB)
        >>> lat, lon = pointC.latitude_deg, pointC.longitude_deg
        >>> msg = 'Ex9, Intersection: lat, long = {:4.2f}, {:4.2f} deg'
        >>> msg.format(lat[0], lon[0])
        'Ex9, Intersection: lat, long = 40.32, 55.90 deg'

    See also Example 9 at http://www.navlab.net/nvector/#example_9

    **Example 10: "Cross track distance"**

    .. image http://www.navlab.net/images/ex10img.png

    Path A is given by the two positions A1 and A2.

    Find the cross track distance sxt between the path A (i.e. the great circle
    through A1 and A2) and the position B (i.e. the shortest distance at the
    surface, between the great circle and B).

    Also find the Euclidean distance dxt between B and the plane defined by the
    great circle. Use Earth radius 6371e3.

    Solution:
        >>> import nvector as nv
        >>> frame = nv.FrameE(a=6371e3, f=0)
        >>> pointA1 = frame.GeoPoint(0, 0, degrees=True)
        >>> pointA2 = frame.GeoPoint(10, 0, degrees=True)
        >>> pointB = frame.GeoPoint(1, 0.1, degrees=True)
        >>> pathA = nv.GeoPath(pointA1, pointA2)
        >>> s_xt = pathA.cross_track_distance(pointB, method='greatcircle').ravel()
        >>> d_xt = pathA.cross_track_distance(pointB, method='euclidean').ravel()
        >>> val_txt = '{:4.2f} km, {:4.2f} km'.format(s_xt[0]/1000, d_xt[0]/1000)
        >>> 'Ex10: Cross track distance: s_xt, d_xt = {}'.format(val_txt)
        'Ex10: Cross track distance: s_xt, d_xt = 11.12 km, 11.12 km'

    See also Example 10 at http://www.navlab.net/nvector/#example_10
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
        # n1 = self.point1.to_nvector()
        # n2 = self.point2.to_nvector()
        # n_EM_E = mean_horizontal_position(np.hstack((n1.normal, n2.normal)))
        # p_EM_E = n1.frame.Nvector(n_EM_E).to_ecef_vector()
        # radius = norm(p_EM_E.pvector, axis=0)
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

        Parameters
        ----------
        method: string
            'greatcircle':
            'euclidean'
        radius: real scalar
            radius of sphere
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
            warnings.warn('Paths are Equal. Intersection point undefined. '
                          'NaN returned.')

        lat_EC, long_EC = n_E2lat_lon(n_EC_E, frame.R_Ee)
        return GeoPoint(lat_EC, long_EC, frame=frame)

    def interpolate(self, ti):
        """
        Return the interpolated point along the path

        Parameters
        ----------
        ti: real scalar
            interpolation time assuming point1 is at t0=0 and point2 is at t1=1

        Returns
        -------
        point: Nvector
            point of interpolation along path
        """
        n_EB_E_t0, n_EB_E_t1 = self._nvectors()

        n_EB_E_ti = unit(n_EB_E_t0 + ti * (n_EB_E_t1 - n_EB_E_t0))
        zi = self.point1.z + ti * (self.point2.z-self.point1.z)
        frame = self.point1.frame
        return frame.Nvector(n_EB_E_ti, zi)

if __name__ == '__main__':
    pass
