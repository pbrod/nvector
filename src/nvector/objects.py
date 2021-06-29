"""
Object oriented interface to geodesic functions
===============================================

"""
# pylint: disable=invalid-name
from __future__ import division, print_function
from functools import partial
import warnings
import numpy as np
from numpy.linalg import norm
from geographiclib.geodesic import Geodesic as _Geodesic
from nvector import _examples, license as _license
from nvector._common import test_docstrings, use_docstring_from, use_docstring, _make_summary
from nvector.rotation import zyx2R, n_E_and_wa2R_EL, n_E2R_EN
from nvector.util import unit, mdot, get_ellipsoid, rad, deg, isclose, allclose, array_to_list_dict
from nvector.core import (lat_lon2n_E,
                          n_E2lat_lon,
                          n_EB_E2p_EB_E,
                          p_EB_E2n_EB_E,
                          closest_point_on_great_circle,
                          course_over_ground,
                          great_circle_distance,
                          euclidean_distance,
                          cross_track_distance,
                          intersect,
                          n_EA_E_distance_and_azimuth2n_EB_E,
                          E_rotation,
                          on_great_circle_path,
                          _interp_vectors)


__all__ = ['delta_E', 'delta_L', 'delta_N',
           'diff_positions',
           'FrameB', 'FrameE', 'FrameN', 'FrameL',
           'GeoPath',
           'GeoPoint',
           'ECEFvector',
           'Nvector',
           'Pvector']


@use_docstring(_examples.get_examples_no_header([1]))
def delta_E(point_a, point_b):
    """
    Returns cartesian delta vector from positions a to b decomposed in E.

    Parameters
    ----------
    point_a, point_b: Nvector, GeoPoint or ECEFvector objects
        position a and b, decomposed in E.

    Returns
    -------
    p_ab_E:  ECEFvector
        Cartesian position vector(s) from a to b, decomposed in E.

    Notes
    -----
    The calculation is exact, taking the ellipsity of the Earth into account.
    It is also non-singular as both n-vector and p-vector are non-singular
    (except for the center of the Earth).

    Examples
    --------
    {super}

    See also
    --------
    n_EA_E_and_p_AB_E2n_EB_E, p_EB_E2n_EB_E, n_EB_E2p_EB_E
    """
    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = point_a.to_ecef_vector()
    p_EB_E = point_b.to_ecef_vector()
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


diff_positions = np.deprecate(delta_E, old_name='diff_positions', new_name='delta_E')


def _base_angle(angle_rad):
    r"""Returns angle so it is between $-\pi$ and $\pi$"""
    return np.mod(angle_rad + np.pi, 2*np.pi) - np.pi


def delta_N(point_a, point_b):
    """Returns cartesian delta vector from positions a to b decomposed in N.

    Parameters
    ----------
    point_a, point_b: Nvector, GeoPoint or ECEFvector objects
        position a and b, decomposed in E.

    See also
    --------
    delta_E, delta_L
    """
    # p_ab_E = delta_E(point_a, point_b)
    # p_ab_N = p_ab_E.change_frame(....)
    return delta_E(point_a, point_b).change_frame(FrameN(point_a))


def _delta(self, other):
    """Returns cartesian delta vector from positions a to b decomposed in N."""
    return delta_N(self, other)


def delta_L(point_a, point_b, wander_azimuth=0):
    """Returns cartesian delta vector from positions a to b decomposed in L.

    Parameters
    ----------
    point_a, point_b: Nvector, GeoPoint or ECEFvector objects
        position a and b, decomposed in E.
    wander_azimuth: real scalar
        Angle [rad] between the x-axis of L and the north direction.

    See also
    --------
    delta_E, delta_N
    """
    local_frame = FrameL(point_a, wander_azimuth=wander_azimuth)
    # p_ab_E = delta_E(point_a, point_b)
    # p_ab_L = p_ab_E.change_frame(....)
    return delta_E(point_a, point_b).change_frame(local_frame)


class _Common(object):
    _NAMES = ()

    def __repr__(self):
        cname = self.__class__.__name__
        fmt = ', '
        names = self._NAMES if self._NAMES else list(self.__dict__)
        dict_params = array_to_list_dict(self.__dict__.copy())
        if 'nvector' in dict_params:
            dict_params['point'] = dict_params['nvector']
        params = fmt.join(['{}={!r}'.format(name, dict_params[name])
                           for name in names if not name.startswith('_')])

        return '{}({})'.format(cname, params)

    def __eq__(self, other):
        try:
            return self is other or self._is_equal_to(other, rtol=1e-12, atol=1e-14)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class GeoPoint(_Common):
    """
    Geographical position given as latitude, longitude, depth in frame E.

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
    >>> point_a = wgs84.GeoPoint(-41.32, 174.81, degrees=True)
    >>> point_b = wgs84.GeoPoint(40.96, -5.50, degrees=True)

    >>> print(point_a)
    GeoPoint(latitude=-0.721170046924057,
             longitude=3.0510100654112877,
             z=0,
            frame=FrameE(a=6378137.0,
                         f=0.0033528106647474805,
                         name='WGS84',
                         axes='e'))

    The geodesic inverse problem

    >>> s12, az1, az2 = point_a.distance_and_azimuth(point_b, degrees=True)
    >>> 's12 = {:5.2f}, az1 = {:5.2f}, az2 = {:5.2f}'.format(s12, az1, az2)
    's12 = 19959679.27, az1 = 161.07, az2 = 18.83'

    The geodesic direct problem

    >>> point_a = wgs84.GeoPoint(40.6, -73.8, degrees=True)
    >>> az1, distance = 45, 10000e3
    >>> point_b, az2 = point_a.displace(distance, az1, degrees=True)
    >>> lat2, lon2 = point_b.latitude_deg, point_b.longitude_deg
    >>> msg = 'lat2 = {:5.2f}, lon2 = {:5.2f}, az2 = {:5.2f}'
    >>> msg.format(lat2, lon2, az2)
    'lat2 = 32.64, lon2 = 49.01, az2 = 140.37'

    """
    _NAMES = ('latitude', 'longitude', 'z', 'frame')

    def __init__(self, latitude, longitude, z=0, frame=None, degrees=False):
        if degrees:
            latitude, longitude = rad(latitude, longitude)
        self.latitude, self.longitude, self.z = np.broadcast_arrays(latitude, longitude, z)
        self.frame = _default_frame(frame)

    def _is_equal_to(self, other, rtol=1e-12, atol=1e-14):
        def diff(angle1, angle2):
            pi2 = 2 * np.pi
            delta = (angle1 - angle2) % pi2
            return np.where(delta > np.pi, pi2 - delta, delta)

        options = dict(rtol=rtol, atol=atol)
        delta_lat = diff(self.latitude, other.latitude)
        delta_lon = diff(self.longitude, other.longitude)
        return (allclose(delta_lat, 0, **options)
                and allclose(delta_lon, 0, **options)
                and allclose(self.z, other.z, **options)
                and self.frame == other.frame)

    @property
    def latlon_deg(self):
        """(latitude_deg, longitude_deg, z) tuple, angles are in degree."""
        return self.latitude_deg, self.longitude_deg, self.z

    @property
    def latlon(self):
        """(latitude, longitude, z) tuple, angles are in radian."""
        return self.latitude, self.longitude, self.z

    @property
    def latitude_deg(self):
        """latitude in degrees."""
        return deg(self.latitude)

    @property
    def longitude_deg(self):
        """longitude in degrees."""
        return deg(self.longitude)

    @property
    def scalar(self):
        """True if the position is a scalar point"""
        return (np.ndim(self.z) == 0
                and np.size(self.latitude) == 1
                and np.size(self.longitude) == 1)

    def to_ecef_vector(self):
        """
        Returns position as ECEFvector object.

        See also
        --------
        ECEFvector
        """
        return self.to_nvector().to_ecef_vector()

    def to_geo_point(self):
        """
        Returns position as GeoPoint object.

        See also
        --------
        GeoPoint
        """

        return self

    def to_nvector(self):
        """
        Returns position as Nvector object.

        See also
        --------
        Nvector
        """
        latitude, longitude = self.latitude, self.longitude
        n_vector = lat_lon2n_E(latitude, longitude, self.frame.R_Ee)
        return Nvector(n_vector, self.z, self.frame)

    delta_to = _delta

    def _displace_great_circle(self, distance, azimuth, degrees):
        """ Returns the great circle solution using the nvector method.
        """
        n_a = self.to_nvector()
        e_a = n_a.to_ecef_vector()
        radius = e_a.length
        distance_rad = distance / radius
        azimuth_rad = azimuth if not degrees else rad(azimuth)
        normal_b = n_EA_E_distance_and_azimuth2n_EB_E(n_a.normal, distance_rad, azimuth_rad)
        point_b = Nvector(normal_b, self.z, self.frame).to_geo_point()
        azimuth_b = _base_angle(delta_N(point_b, e_a).azimuth - np.pi)
        if degrees:
            return point_b, deg(azimuth_b)
        return point_b, azimuth_b

    def displace(self, distance, azimuth, long_unroll=False, degrees=False, method='ellipsoid'):
        """
        Returns position b computed from current position, distance and azimuth.

        Parameters
        ----------
        distance: real scalar
            ellipsoidal or great circle distance [m] between position A and B.
        azimuth:
            azimuth [rad or deg] of line at position A.
        long_unroll: bool
            Controls the treatment of longitude when method=='ellipsoid'.
            See distance_and_azimuth method for details.
        degrees: bool
            azimuths are given in degrees if True otherwise in radians.
        method: 'greatcircle' or 'ellipsoid'
            defining the path where to find position b.

        Returns
        -------
        point_b:  GeoPoint object
            latitude and longitude of position B.
        azimuth_b
            azimuth [rad or deg] of line at position B.

        """
        if method[:1] == 'e':  # exact solution
            return self._displace_ellipsoid(distance, azimuth, long_unroll, degrees)
        return self._displace_great_circle(distance, azimuth, degrees)

    def _displace_ellipsoid(self, distance, azimuth, long_unroll=False, degrees=False):
        """ Returns the exact ellipsoidal solution using the method of Karney.
        """
        frame = self.frame
        z = self.z
        if not degrees:
            azimuth = deg(azimuth)
        lat_a, lon_a = self.latitude_deg, self.longitude_deg
        lat_b, lon_b, azimuth_b = frame.direct(lat_a, lon_a, azimuth, distance,
                                               z=z, long_unroll=long_unroll,
                                               degrees=True)

        point_b = frame.GeoPoint(latitude=lat_b, longitude=lon_b, z=z, degrees=True)
        if not degrees:
            return point_b, rad(azimuth_b)
        return point_b, azimuth_b

    def distance_and_azimuth(self, point, long_unroll=False, degrees=False, method='ellipsoid'):
        """
        Returns ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        point:  GeoPoint object
            Latitude and longitude of position b.
        long_unroll: bool
            Controls the treatment of longitude. If it is False then the lon_a and
            lon_b are both reduced to the range [-180, 180). If it is True, then
            lon_a is as given in the function call and (lon_b - lon_a) determines
            how many times and in what sense the geodesic has encircled the ellipsoid.
        degrees: bool
            azimuths are returned in degrees if True otherwise in radians.
        method: 'greatcircle' or 'ellipsoid'
            defining the path distance.

        Returns
        -------
        s_ab: real scalar or vector of length n.
            ellipsoidal distance [m] between position a and b at their average height.
        azimuth_a, azimuth_b: real scalars or vectors of length n.
            direction [rad or deg] of line at position a and b relative to
            North, respectively.

        Notes
        -----
        Restriction on the parameters:
        * Latitudes must lie between -90 and 90 degrees.
        * Latitudes outside this range will be set to NaNs.
        * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        Examples
        --------
        >>> import nvector as nv
        >>> point1 = nv.GeoPoint(0, 0)
        >>> point2 = nv.GeoPoint(0.5, 179.5, degrees=True)
        >>> s_12, az1, azi2 = point1.distance_and_azimuth(point2)
        >>> nv.allclose(s_12, 19936288.579)
        True

        References
        ----------
        `C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55 (2013)
        <https://rdcu.be/cccgm>`_

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_
        """
        _check_frames(self, point)
        if method[0] == 'e':
            return self._distance_and_azimuth_ellipsoid(point, long_unroll, degrees)
        return self._distance_and_azimuth_greatcircle(point, degrees)

    def _distance_and_azimuth_greatcircle(self, point, degrees):
        n_a = self.to_nvector()
        n_b = point.to_nvector()
        e_a = n_a.to_ecef_vector()
        e_b = n_b.to_ecef_vector()
        radius = 0.5 * (e_a.length + e_b.length)
        distance = great_circle_distance(n_a.normal, n_b.normal, radius)
        azimuth_a = delta_N(e_a, e_b).azimuth
        azimuth_b = _base_angle(delta_N(e_b, e_a).azimuth - np.pi)

        if degrees:
            azimuth_a, azimuth_b = deg(azimuth_a), deg(azimuth_b)

        if np.ndim(radius) == 0:
            return distance[0], azimuth_a, azimuth_b  # scalar track distance
        return distance, azimuth_a, azimuth_b

    def _distance_and_azimuth_ellipsoid(self, point, long_unroll, degrees):
        gpoint = point.to_geo_point()
        lat_a, lon_a = self.latitude, self.longitude
        lat_b, lon_b = gpoint.latitude, gpoint.longitude
        z = 0.5 * (self.z + gpoint.z)  # Average depth

        if degrees:
            lat_a, lon_a, lat_b, lon_b = deg(lat_a, lon_a, lat_b, lon_b)

        return self.frame.inverse(lat_a, lon_a, lat_b, lon_b, z, long_unroll, degrees)


class Nvector(_Common):
    """
    Geographical position(s) given as n-vector(s) and depth(s) in frame E

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

    Examples
    --------
    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> point_a = wgs84.GeoPoint(-41.32, 174.81, degrees=True)
    >>> point_b = wgs84.GeoPoint(40.96, -5.50, degrees=True)
    >>> nv_a = point_a.to_nvector()
    >>> print(nv_a)
    Nvector(normal=[[-0.7479546170813224], [0.06793758070955484], [-0.6602638683996461]],
            z=0,
            frame=FrameE(a=6378137.0,
                        f=0.0033528106647474805,
                        name='WGS84',
                        axes='e'))


    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """

    _NAMES = ('normal', 'z', 'frame')

    def __init__(self, normal, z=0, frame=None):
        self.normal = normal
        self.z = z
        self.frame = _default_frame(frame)

    def interpolate(self, t_i, t, kind='linear', window_length=0, polyorder=2,
                    mode='interp', cval=0.0):
        """
        Returns interpolated values from nvector data.

        Parameters
        ----------
        t_i: real vector length m
            Vector of interpolation times.
        t: real vector length n
            Vector of times.
        kind: str or int, optional
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.
        window_length: positive odd integer
            The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
            Default window_length=0, i.e. no smoothing.
        polyorder: int
            The order of the polynomial used to fit the samples.
            polyorder must be less than window_length.
        mode: 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
            Determines the type of extension to use for the padded signal to
            which the filter is applied.  When mode is 'constant', the padding
            value is given by cval.
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree polyorder polynomial is fit to the
            last window_length values of the edges, and this polynomial is
            used to evaluate the last window_length // 2 output values.
        cval: scalar, optional
            Value to fill past the edges of the input if mode is 'constant'.
            Default is 0.0.

        Returns
        -------
        result: Nvector objest
            Interpolated n-vector(s) [no unit] decomposed in E.

        Notes
        -----
        The result for spherical Earth is returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import nvector as nv
        >>> lat = np.arange(0, 10)
        >>> lon = nv.deg(np.sin(nv.rad(np.linspace(-90, 70, 10))))
        >>> nvectors = nv.GeoPoint(lat, lon, degrees=True).to_nvector()
        >>> t = np.arange(10)
        >>> t_i = np.linspace(0, t[-1], 100)
        >>> nvectors_i = nvectors.interpolate(t_i, t, kind='cubic')
        >>> lati, loni, zi = nvectors_i.to_geo_point().latlon_deg
        >>> h = plt.plot(lon, lat, 'o', loni, lati, '-')
        >>> plt.show() # doctest: +SKIP
        >>> plt.close()
        """
        vectors = np.vstack((self.normal, self.z))
        vectors_i = _interp_vectors(t_i, t, vectors, kind, window_length, polyorder, mode, cval)
        normal = unit(vectors_i[:3], norm_zero_vector=np.nan)
        return Nvector(normal, z=vectors_i[3], frame=self.frame)

    def to_ecef_vector(self):
        """
        Returns position as ECEFvector object.

        See also
        --------
        ECEFvector
        """
        frame = self.frame
        a, f, R_Ee = frame.a, frame.f, frame.R_Ee
        pvector = n_EB_E2p_EB_E(self.normal, depth=self.z, a=a, f=f, R_Ee=R_Ee)
        scalar = self.scalar
        return ECEFvector(pvector, self.frame, scalar=scalar)

    @property
    def scalar(self):
        """True if the position is a scalar point"""
        return np.ndim(self.z) == 0 and self.normal.shape[1] == 1

    def to_geo_point(self):
        """
        Returns position as GeoPoint object.

        See also
        --------
        GeoPoint
        """
        latitude, longitude = n_E2lat_lon(self.normal, R_Ee=self.frame.R_Ee)

        if self.scalar:
            return GeoPoint(latitude[0], longitude[0], self.z, self.frame)  # Scalar geo_point
        return GeoPoint(latitude, longitude, self.z, self.frame)

    def to_nvector(self):
        """
        Returns position as Nvector object.

        See also
        --------
        Nvector
        """
        return self

    delta_to = _delta

    def unit(self):
        """Normalizes self to unit vector(s)"""
        self.normal = unit(self.normal)

    def course_over_ground(self, window_length=0, polyorder=2, mode='nearest', cval=0.0):
        """Returns course over ground in radians from nvector positions

        Parameters
        ----------
        window_length: positive odd integer
            The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
            Default window_length=0, i.e. no smoothing.
        polyorder: int
            The order of the polynomial used to fit the samples.
            polyorder must be less than window_length.
        mode: 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
            Determines the type of extension to use for the padded signal to
            which the filter is applied.  When mode is 'constant', the padding
            value is given by cval. When the 'nearest' mode is selected (the default)
            the extension contains the nearest input value.
            When the 'interp' mode is selected, no extension
            is used.  Instead, a degree polyorder polynomial is fit to the
            last window_length values of the edges, and this polynomial is
            used to evaluate the last window_length // 2 output values.
        cval: scalar, optional
            Value to fill past the edges of the input if mode is 'constant'.
            Default is 0.0.

        Returns
        -------
        cog: numpy array
            angle in radians clockwise from True North to the direction towards
            which the vehicle travels.

        Notes
        -----
        Please be aware that this method requires the vehicle positions to be very smooth!
        If they are not you should probably smooth it by a window_length corresponding
        to a few seconds or so.

        See https://www.navlab.net/Publications/The_Seven_Ways_to_Find_Heading.pdf
        for an overview of methods to find accurate headings.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import nvector as nv
        >>> points = nv.GeoPoint((59.381509, 59.387647),(10.496590, 10.494713), degrees=True)
        >>> nvec = points.to_nvector()
        >>> COG_rad = nvec.course_over_ground()
        >>> dx, dy = np.sin(COG_rad[0]), np.cos(COG_rad[0])
        >>> COG = nv.deg(COG_rad)
        >>> p_AB_N = nv.n_EA_E_and_n_EB_E2p_AB_N(nvec.normal[:, :1], nvec.normal[:, 1:]).ravel()
        >>> ax = plt.figure().gca()
        >>> _ = ax.plot(0, 0, 'bo', label='A')
        >>> _ = ax.arrow(0,0, dx*300, dy*300, head_width=20)
        >>> _ = ax.plot(p_AB_N[1], p_AB_N[0], 'go', label='B')
        >>> _ = ax.set_title(f'COG={COG} degrees')
        >>> _ = ax.set_xlabel('East [m]')
        >>> _ = ax.set_ylabel('North [m]')
        >>> _ = ax.set_xlim(-500, 200)
        >>> _ = ax.set_aspect('equal', adjustable='box')
        >>> _ = ax.legend()
        >>> plt.show() # doctest + SKIP
        >>> plt.close()
        """
        frame = self.frame
        options = dict(a=frame.a, f=frame.f, R_Ee=frame.R_Ee)
        return course_over_ground(self.normal, window_length, polyorder, mode, cval, **options)

    def mean(self):
        """
        Returns mean position of the n-vectors.
        """
        average_nvector = unit(np.sum(self.normal, axis=1).reshape((3, 1)))
        return self.frame.Nvector(average_nvector, z=np.mean(self.z))

    mean_horizontal_position = np.deprecate(mean,
                                            old_name='mean_horizontal_position',
                                            new_name='mean')

    def _is_equal_to(self, other, rtol=1e-12, atol=1e-14):
        options = dict(rtol=rtol, atol=atol)
        return (allclose(self.normal, other.normal, **options)
                and allclose(self.z, other.z, **options)
                and self.frame == other.frame)

    def __add__(self, other):
        _check_frames(self, other)
        return self.frame.Nvector(self.normal + other.normal, self.z + other.z)

    def __sub__(self, other):
        _check_frames(self, other)
        return self.frame.Nvector(self.normal - other.normal, self.z - other.z)

    def __neg__(self):
        return self.frame.Nvector(-self.normal, -self.z)

    def __mul__(self, scalar):
        """elementwise multiplication"""

        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal * scalar, self.z * scalar)
        return NotImplemented  # 'Only scalar multiplication is implemented'

    def __div__(self, scalar):
        """elementwise division"""
        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal / scalar, self.z / scalar)
        return NotImplemented  # 'Only scalar division is implemented'

    __truediv__ = __div__
    __radd__ = __add__
    __rmul__ = __mul__


class Pvector(_Common):
    """
    Geographical position given as cartesian position vector in a frame.
    """

    _NAMES = ('pvector', 'frame', 'scalar')

    def __init__(self, pvector, frame, scalar=None):
        if scalar is None:
            scalar = np.shape(pvector)[1] == 1
        self.pvector = pvector
        self.frame = frame
        self.scalar = scalar

    def _is_equal_to(self, other, rtol=1e-12, atol=1e-14):
        options = dict(rtol=rtol, atol=atol)
        return (allclose(self.pvector, other.pvector, **options)
                and self.frame == other.frame)

    def to_ecef_vector(self):
        """
        Returns position as ECEFvector object.

        See also
        --------
        ECEFvector
        """

        n_frame = self.frame
        p_AB_N = self.pvector
        # alternatively: np.dot(n_frame.R_EN, p_AB_N)
        p_AB_E = mdot(n_frame.R_EN, p_AB_N[:, None, ...]).reshape(3, -1)
        return ECEFvector(p_AB_E, frame=n_frame.nvector.frame, scalar=self.scalar)

    def to_nvector(self):
        """
        Returns position as Nvector object.

        See also
        --------
        Nvector
        """

        return self.to_ecef_vector().to_nvector()

    def to_geo_point(self):
        """
        Returns position as GeoPoint object.

        See also
        --------
        GeoPoint
        """
        return self.to_ecef_vector().to_geo_point()

    delta_to = _delta

    @property
    def length(self):
        "Length of the pvector."
        lengths = norm(self.pvector, axis=0)
        if self.scalar:
            return lengths[0]
        return lengths

    @property
    def azimuth_deg(self):
        """Azimuth in degree clockwise relative to the x-axis."""
        return deg(self.azimuth)

    @property
    def azimuth(self):
        """Azimuth in radian clockwise relative to the x-axis."""
        p_AB_N = self.pvector
        if self.scalar:
            return np.arctan2(p_AB_N[1], p_AB_N[0])[0]
        return np.arctan2(p_AB_N[1], p_AB_N[0])

    @property
    def elevation_deg(self):
        """Elevation in degree relative to the xy-plane. (Positive downwards in a NED frame)"""
        return deg(self.elevation)

    @property
    def elevation(self):
        """Elevation in radian relative to the xy-plane. (Positive downwards in a NED frame)"""
        z = self.pvector[2]
        if self.scalar:
            return np.arcsin(z / self.length)[0]
        return np.arcsin(z / self.length)


@use_docstring(_examples.get_examples_no_header([3, 4]))
class ECEFvector(Pvector):
    """
    Geographical position given as cartesian position vector in frame E

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

    Examples
    --------
    {super}

    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """

    def __init__(self, pvector, frame=None, scalar=None):
        super(ECEFvector, self).__init__(pvector, _default_frame(frame), scalar)

    def change_frame(self, frame):
        """
        Converts to Cartesian position vector in another frame

        Parameters
        ----------
        frame: FrameB, FrameN or frameL object
            local frame M used to convert p_AB_E (position vector from A to B,
            decomposed in E) to a cartesian vector p_AB_M decomposed in M.

        Returns
        -------
        p_AB_M:  Pvector object
            position vector from A to B, decomposed in frame M.

        See also
        --------
        n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
        """
        _check_frames(self, frame.nvector)
        p_AB_E = self.pvector
        p_AB_N = mdot(np.swapaxes(frame.R_EN, 1, 0), p_AB_E[:, None, ...])
        return Pvector(p_AB_N.reshape(3, -1), frame=frame, scalar=self.scalar)

    def to_ecef_vector(self):
        return self

    def to_geo_point(self):
        """
        Returns position as GeoPoint object.

        See also
        --------
        GeoPoint
        """
        return self.to_nvector().to_geo_point()

    def to_nvector(self):
        """
        Returns position as Nvector object.

        See also
        --------
        Nvector
        """
        frame = self.frame
        p_EB_E = self.pvector
        R_Ee = frame.R_Ee
        n_EB_E, depth = p_EB_E2n_EB_E(p_EB_E, a=frame.a, f=frame.f, R_Ee=R_Ee)
        if self.scalar:
            return Nvector(n_EB_E, z=depth[0], frame=frame)
        return Nvector(n_EB_E, z=depth, frame=frame)

    delta_to = _delta

    def __add__(self, other):
        _check_frames(self, other)
        scalar = self.scalar and other.scalar
        return ECEFvector(self.pvector + other.pvector, self.frame, scalar)

    def __sub__(self, other):
        _check_frames(self, other)
        scalar = self.scalar and other.scalar
        return ECEFvector(self.pvector - other.pvector, self.frame, scalar)

    def __neg__(self):
        return ECEFvector(-self.pvector, self.frame, self.scalar)


@use_docstring(_examples.get_examples_no_header([5, 6, 9, 10]))
class GeoPath(object):
    """
    Geographical path between two positions in Frame E

    Parameters
    ----------
     point_a, point_b: Nvector, GeoPoint or ECEFvector objects
        The path is defined by the line between position A and B, decomposed
        in E.

    Notes
    -----
    Please note that either position A or B or both might be a vector of points.
    In this case the GeoPath instance represents all the paths between the positions
    of A and the corresponding positions of B.

    Examples
    --------
    {super}
    """

    def __init__(self, point_a, point_b):
        self.point_a = point_a
        self.point_b = point_b

    @property
    def positionA(self):
        """positionA is deprecated, use point_a instead!"""  # @ReservedAssignment
        warnings.warn("positionA is deprecated, use point_a instead!",
                      category=DeprecationWarning, stacklevel=2)
        return self.point_a

    @property
    def positionB(self):
        """positionB is deprecated, use point_b instead!"""  # @ReservedAssignment
        warnings.warn("positionB is deprecated, use point_b instead!",
                      category=DeprecationWarning, stacklevel=2)
        return self.point_b

    def nvectors(self):
        """ Returns point_a and point_b as n-vectors
        """
        return self.point_a.to_nvector(), self.point_b.to_nvector()

    def geo_points(self):
        """ Returns point_a and point_b as geo-points
        """
        return self.point_a.to_geo_point(), self.point_b.to_geo_point()

    def ecef_vectors(self):
        """ Returns point_a and point_b as  ECEF-vectors
        """
        return self.point_a.to_ecef_vector(), self.point_b.to_ecef_vector()

    def nvector_normals(self):
        """Returns nvector normals for position a and b"""
        nvector_a, nvector_b = self.nvectors()
        return nvector_a.normal, nvector_b.normal

    def _get_average_radius(self):
        p_E1_E, p_E2_E = self.ecef_vectors()
        radius = (p_E1_E.length + p_E2_E.length) / 2
        return radius

    def cross_track_distance(self, point, method='greatcircle', radius=None):
        """
        Returns cross track distance from path to point.

        Parameters
        ----------
        point: GeoPoint, Nvector or ECEFvector object
            position to measure the cross track distance to.
        radius: real scalar
            radius of sphere in [m]. Default is the average height of points A and B.
        method: 'greatcircle' or 'euclidean'
            defining distance calculated.

        Returns
        -------
        distance: real scalar or vector
            distance in [m]

        Notes
        -----
        The result for spherical Earth is returned.
        """
        if radius is None:
            radius = self._get_average_radius()
        path = self.nvector_normals()
        n_c = point.to_nvector().normal
        distance = cross_track_distance(path, n_c, method=method, radius=radius)
        if np.ndim(radius) == 0 and distance.size == 1:
            return distance[0]  # scalar cross track distance
        return distance

    def track_distance(self, method='greatcircle', radius=None):
        """
        Returns the path distance computed at the average height.

        Parameters
        ----------
        method:  'greatcircle',  'euclidean' or 'ellipsoidal'
            defining distance calculated.
        radius: real scalar
            radius of sphere. Default is the average height of points A and B
        """
        if method[:2] in {'ex', 'el'}:  # exact or ellipsoidal
            point_a, point_b = self.geo_points()
            s_ab, _angle1, _angle2 = point_a.distance_and_azimuth(point_b)
            return s_ab
        if radius is None:
            radius = self._get_average_radius()
        normal_a, normal_b = self.nvector_normals()

        distance_fun = euclidean_distance if method[:2] == "eu" else great_circle_distance
        distance = distance_fun(normal_a, normal_b, radius)
        if np.ndim(radius) == 0:
            return distance[0]  # scalar track distance
        return distance

    def intersect(self, path):
        """
        Returns the intersection(s) between the great circles of the two paths

        Parameters
        ----------
        path: GeoPath object
            path to intersect

        Returns
        -------
        point: GeoPoint
            point of intersection between paths

        Notes
        -----
        The result for spherical Earth is returned at the average height.
        """
        frame = self.point_a.frame
        point_a1, point_a2 = self.nvectors()
        point_b1, point_b2 = path.nvectors()
        path_a = (point_a1.normal, point_a2.normal)  # self.nvector_normals()
        path_b = (point_b1.normal, point_b2.normal)  # path.nvector_normals()
        normal_c = intersect(path_a, path_b)  # nvector
        depth = (point_a1.z + point_a2.z + point_b1.z + point_b2.z) / 4.
        return frame.Nvector(normal_c, z=depth)

    intersection = np.deprecate(intersect,
                                old_name='intersection',
                                new_name='intersect')

    def _on_ellipsoid_path(self, point, rtol=1e-6, atol=1e-8):
        point_a, point_b = self.geo_points()
        point_c = point.to_geo_point()
        z = (point_a.z + point_b.z) * 0.5
        distance_ab, azimuth_ab, _azi_ba = point_a.distance_and_azimuth(point_b)
        distance_ac, azimuth_ac, _azi_ca = point_a.distance_and_azimuth(point_c)
        return (isclose(z, point_c.z, rtol=rtol, atol=atol)
                & (isclose(distance_ac, 0, atol=atol)
                   | ((distance_ab >= distance_ac)
                      & isclose(azimuth_ac, azimuth_ab, rtol=rtol, atol=atol))))

    def on_great_circle(self, point, atol=1e-8):
        """Returns True if point is on the great circle within a tolerance."""
        distance = np.abs(self.cross_track_distance(point))
        result = isclose(distance, 0, atol=atol)
        if np.ndim(result) == 0:
            return result[()]
        return result

    def _on_great_circle_path(self, point, radius=None, rtol=1e-9, atol=1e-8):
        if radius is None:
            radius = self._get_average_radius()
        n_a, n_b = self.nvectors()
        path = (n_a.normal, n_b.normal)
        n_c = point.to_nvector()
        same_z = isclose(n_c.z, (n_a.z + n_b.z) * 0.5, rtol=rtol, atol=atol)
        result = on_great_circle_path(path, n_c.normal, radius, atol=atol) & same_z
        if np.ndim(radius) == 0 and result.size == 1:
            return result[0]  # scalar outout
        return result

    def on_path(self, point, method='greatcircle', rtol=1e-6, atol=1e-8):
        """
        Returns True if point is on the path between A and B witin a tolerance.

        Parameters
        ----------
        point : Nvector, GeoPoint or ECEFvector
            point to test
        method: 'greatcircle' or 'ellipsoid'
            defining the path.

        Returns
        -------
        result: Bool scalar or boolean vector
            True if the point is on the path at its average height.

        Notes
        -----
        The result for spherical Earth is returned for method='greatcircle'.

        Examples
        --------
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> pointA = wgs84.GeoPoint(89, 0, degrees=True)
        >>> pointB = wgs84.GeoPoint(80, 0, degrees=True)
        >>> path = nv.GeoPath(pointA, pointB)
        >>> pointC = path.interpolate(0.6).to_geo_point()
        >>> path.on_path(pointC)
        True
        >>> path.on_path(pointC, 'ellipsoid')
        True
        >>> pointD = path.interpolate(1.000000001).to_geo_point()
        >>> path.on_path(pointD)
        False
        >>> path.on_path(pointD, 'ellipsoid')
        False
        >>> pointE = wgs84.GeoPoint(85, 0.0001, degrees=True)
        >>> path.on_path(pointE)
        False
        >>> pointC = path.interpolate(-2).to_geo_point()
        >>> path.on_path(pointC)
        False
        >>> path.on_great_circle(pointC)
        True
        """
        if method[:2] in {'ex', 'el'}:  # exact or ellipsoid
            return self._on_ellipsoid_path(point, rtol=rtol, atol=atol)
        return self._on_great_circle_path(point, rtol=rtol, atol=atol)

    def _closest_point_on_great_circle(self, point):
        point_c = point.to_nvector()
        point_a, point_b = self.nvectors()
        path = (point_a.normal, point_b.normal)
        z = (point_a.z + point_b.z) * 0.5
        normal_d = closest_point_on_great_circle(path, point_c.normal)
        return point_c.frame.Nvector(normal_d, z)

    def closest_point_on_great_circle(self, point):
        """
        Returns closest point on great circle path to the point.

        Parameters
        ----------
        point: GeoPoint
            point of intersection between paths

        Returns
        -------
        closest_point: GeoPoint
            closest point on path.

        Notes
        -----
        The result for spherical Earth is returned at the average depth.

        Examples
        --------
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> point_a = wgs84.GeoPoint(51., 1., degrees=True)
        >>> point_b = wgs84.GeoPoint(51., 2., degrees=True)
        >>> point_c = wgs84.GeoPoint(51., 2.9, degrees=True)
        >>> path = nv.GeoPath(point_a, point_b)
        >>> point = path.closest_point_on_great_circle(point_c)
        >>> path.on_path(point)
        False
        >>> nv.allclose((point.latitude_deg, point.longitude_deg),
        ...             (50.99270338, 2.89977984))
        True

        >>> nv.allclose(GeoPath(point_c, point).track_distance(),  810.76312076)
        True

        """

        point_d = self._closest_point_on_great_circle(point)
        return point_d.to_geo_point()

    def closest_point_on_path(self, point):
        """
        Returns closest point on great circle path segment to the point.

        If the point is within the extent of the segment, the point returned is
        on the segment path otherwise, it is the closest endpoint defining the
        path segment.

        Parameters
        ----------
        point: GeoPoint
            point of intersection between paths

        Returns
        -------
        closest_point: GeoPoint
            closest point on path segment.

        Examples
        --------
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name='WGS84')
        >>> pointA = wgs84.GeoPoint(51., 1., degrees=True)
        >>> pointB = wgs84.GeoPoint(51., 2., degrees=True)
        >>> pointC = wgs84.GeoPoint(51., 1.9, degrees=True)
        >>> path = nv.GeoPath(pointA, pointB)
        >>> point = path.closest_point_on_path(pointC)
        >>> np.allclose((point.latitude_deg, point.longitude_deg),
        ...             (51.00038411380564, 1.900003311624411))
        True
        >>> np.allclose(GeoPath(pointC, point).track_distance(),  42.67368351)
        True
        >>> pointD = wgs84.GeoPoint(51.0, 2.1, degrees=True)
        >>> pointE = path.closest_point_on_path(pointD) # 51.0000, 002.0000
        >>> pointE.latitude_deg, pointE.longitude_deg
        (51.0, 2.0)
        """
        # TODO: vectorize this
        return self._closest_point_on_path(point)

    def _closest_point_on_path(self, point):
        point_c = self._closest_point_on_great_circle(point)
        if self.on_path(point_c):
            return point_c.to_geo_point()
        n0 = point.to_nvector().normal
        n1, n2 = self.nvector_normals()
        radius = self._get_average_radius()
        d1 = great_circle_distance(n1, n0, radius)
        d2 = great_circle_distance(n2, n0, radius)
        if d1 < d2:
            return self.point_a.to_geo_point()
        return self.point_b.to_geo_point()

    def interpolate(self, ti):
        """
        Returns the interpolated point along the path

        Parameters
        ----------
        ti: real scalar
            interpolation time assuming position A and B is at t0=0 and t1=1,
            respectively.

        Returns
        -------
        point: Nvector
            point of interpolation along path
        """
        point_a, point_b = self.nvectors()
        point_c = point_a + (point_b - point_a) * ti
        point_c.normal = unit(point_c.normal, norm_zero_vector=np.nan)
        return point_c


class FrameE(_Common):

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

    _NAMES = ('a', 'f', 'name', 'axes')

    def __init__(self, a=None, f=None, name='WGS84', axes='e'):
        if a is None or f is None:
            a, f, _full_name = get_ellipsoid(name)
        self.a = a
        self.f = f
        self.name = name
        self.axes = axes

    @property
    def R_Ee(self):
        """Rotation matrix R_Ee defining the axes of the coordinate frame E"""
        return E_rotation(self.axes)

    def _is_equal_to(self, other, rtol=1e-12, atol=1e-14):
        return (allclose(self.a, other.a, rtol=rtol, atol=atol)
                and allclose(self.f, other.f, rtol=rtol, atol=atol)
                and allclose(self.R_Ee, other.R_Ee, rtol=rtol, atol=atol))

    def inverse(self, lat_a, lon_a, lat_b, lon_b, z=0, long_unroll=False, degrees=False):
        """
        Returns ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        lat_a, lon_a:  real scalars or vectors.
            Latitude and longitude of position a.
        lat_b, lon_b:  real scalars or vectors.
            Latitude and longitude of position b.
        z : real scalar or vector
            depth relative to Earth ellipsoid.
        long_unroll: bool
            Controls the treatment of longitude. If it is False then the lon_a and lon_b
            are both reduced to the range [-180, 180). If it is True, then lon_a
            is as given in the function call and (lon_b - lon_a) determines how many times
            and in what sense the geodesic has encircled the ellipsoid.
        degrees: bool
            angles are given in degrees if True otherwise in radians.

        Returns
        -------
        s_ab: real scalar or vector
            ellipsoidal distance [m] between position A and B.
        azimuth_a, azimuth_b:  real scalars or vectors.
            direction [rad or deg] of line at position A and B relative to
            North, respectively.

        Notes
        -----
        Restriction on the parameters:

          * Latitudes must lie between -90 and 90 degrees.
          * Latitudes outside this range will be set to NaNs.
          * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        References
        ----------
        `C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55 (2013)
        <https://rdcu.be/cccgm>`_

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_

        """
        if not degrees:
            lat_a, lon_a, lat_b, lon_b = deg(lat_a, lon_a, lat_b, lon_b)

        lat_a, lon_a, lat_b, lon_b, z = np.broadcast_arrays(lat_a, lon_a, lat_b, lon_b, z)
        fun = partial(self._inverse, outmask=self._outmask(long_unroll))
        items = zip(*np.atleast_1d(lat_a, lon_a, lat_b, lon_b, z))
        sab, azia, azib = np.transpose([fun(lat_ai, lon_ai, lat_bi, lon_bi, z=zi)
                                        for lat_ai, lon_ai, lat_bi, lon_bi, zi in items])

        if not degrees:
            s_ab, azimuth_a, azimuth_b = sab.ravel(), rad(azia.ravel()), rad(azib.ravel())
        else:
            s_ab, azimuth_a, azimuth_b = sab.ravel(), azia.ravel(), azib.ravel()

        if np.ndim(lat_a) == 0:
            return s_ab[0], azimuth_a[0], azimuth_b[0]
        return s_ab, azimuth_a, azimuth_b

    def _inverse(self, lat_a, lon_a, lat_b, lon_b, z=0, outmask=None):
        geo = _Geodesic(self.a - z, self.f)
        result = geo.Inverse(lat_a, lon_a, lat_b, lon_b, outmask=outmask)
        return result['s12'], result['azi1'], result['azi2']

    @staticmethod
    def _outmask(long_unroll):
        if long_unroll:
            return _Geodesic.STANDARD | _Geodesic.LONG_UNROLL
        return _Geodesic.STANDARD

    def direct(self, lat_a, lon_a, azimuth, distance, z=0, long_unroll=False, degrees=False):
        """
        Returns position B computed from position A, distance and azimuth.

        Parameters
        ----------
        lat_a, lon_a:  real scalars or vectors of length n.
            Latitude and longitude [rad or deg] of position A.
        azimuth:  real scalar or vector of length n.
            azimuth [rad or deg] of line at position A relative to North.
        distance: real scalar or vector of length n.
            ellipsoidal distance [m] between position A and B.
        z: real scalar or vector of length n.
            depth relative to Earth ellipsoid.
        long_unroll: bool
            Controls the treatment of longitude. If it is False then the lon_a and lon_b
            are both reduced to the range [-180, 180). If it is True, then lon_a
            is as given in the function call and (lon_b - lon_a) determines how many times
            and in what sense the geodesic has encircled the ellipsoid.
        degrees: bool
            angles are given in degrees if True otherwise in radians.

        Returns
        -------
        lat_b, lon_b:  real scalars or vectors of length n
            Latitude and longitude of position b.
        azimuth_b: real scalar or vector of length n.
            azimuth [rad or deg] of line at position B relative to North.

        Notes
        -----
        Restriction on the parameters:

          * Latitudes must lie between -90 and 90 degrees.
          * Latitudes outside this range will be set to NaNs.
          * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        References
        ----------
        `C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43-55 (2013)
        <https://rdcu.be/cccgm>`_

        `geographiclib <https://pypi.python.org/pypi/geographiclib>`_
        """
        if not degrees:
            lat_a, lon_a, azimuth = deg(lat_a, lon_a, azimuth)

        broadcast = np.broadcast_arrays
        lat_a, lon_a, azimuth, distance, z = broadcast(lat_a, lon_a, azimuth, distance, z)
        fun = partial(self._direct, outmask=self._outmask(long_unroll))

        items = zip(*np.atleast_1d(lat_a, lon_a, azimuth, distance, z))
        lab, lob, azib = np.transpose([fun(lat_ai, lon_ai, azimuthi, distancei, z=zi)
                                       for lat_ai, lon_ai, azimuthi, distancei, zi in items])
        if not degrees:
            latb, lonb, azimuth_b = rad(lab.ravel(), lob.ravel(), azib.ravel())
        else:
            latb, lonb, azimuth_b = lab.ravel(), lob.ravel(), azib.ravel()
        if np.ndim(lat_a) == 0:
            return latb[0], lonb[0], azimuth_b[0]
        return latb, lonb, azimuth_b

    def _direct(self, lat_a, lon_a, azimuth, distance, z=0, outmask=None):
        geo = _Geodesic(self.a - z, self.f)
        result = geo.Direct(lat_a, lon_a, azimuth, distance, outmask=outmask)
        latb, lonb, azimuth_b = result['lat2'], result['lon2'], result['azi2']
        return latb, lonb, azimuth_b

    @use_docstring_from(GeoPoint)
    def GeoPoint(self, *args, **kwds):
        "{super}"
        kwds.pop('frame', None)
        return GeoPoint(*args, frame=self, **kwds)

    @use_docstring_from(Nvector)
    def Nvector(self, *args, **kwds):
        "{super}"
        kwds.pop('frame', None)
        return Nvector(*args, frame=self, **kwds)

    @use_docstring_from(ECEFvector)
    def ECEFvector(self, *args, **kwds):
        "{super}"
        kwds.pop('frame', None)
        return ECEFvector(*args, frame=self, **kwds)


class _LocalFrame(_Common):

    def Pvector(self, pvector):
        """Returns Pvector relative to the local frame."""
        return Pvector(pvector, frame=self)


@use_docstring(_examples.get_examples_no_header([1]))
class FrameN(_LocalFrame):
    """
    North-East-Down frame

    Parameters
    ----------
    point: ECEFvector, GeoPoint or Nvector object
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
    {super}

    See also
    --------
    FrameE, FrameL, FrameB
    """

    _NAMES = ('point',)

    def __init__(self, point):
        nvector = point.to_nvector()
        self.nvector = Nvector(nvector.normal, z=0, frame=nvector.frame)

    @property
    def R_EN(self):
        """Rotation matrix to go between E and N frame"""
        nvector = self.nvector
        return n_E2R_EN(nvector.normal, nvector.frame.R_Ee)

    def _is_equal_to(self, other, rtol=1e-12, atol=1e-14):
        return (allclose(self.R_EN, other.R_EN, rtol=rtol, atol=atol)
                and self.nvector == other.nvector)


class FrameL(FrameN):

    """
    Local level, Wander azimuth frame

    Parameters
    ----------
    point: ECEFvector, GeoPoint or Nvector object
        position of the vehicle (B) which also defines the origin of the local
        frame L. The origin is directly beneath or above the vehicle (B), at
        Earth's surface (surface of ellipsoid model).
    wander_azimuth: real scalar
        Angle [rad] between the x-axis of L and the north direction.

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
    _NAMES = ('point', 'wander_azimuth')

    def __init__(self, point, wander_azimuth=0):
        super(FrameL, self).__init__(point)
        self.wander_azimuth = wander_azimuth

    @property
    def R_EN(self):
        """Rotation matrix to go between E and L frame"""
        n_EA_E = self.nvector.normal
        R_Ee = self.nvector.frame.R_Ee
        return n_E_and_wa2R_EL(n_EA_E, self.wander_azimuth, R_Ee=R_Ee)


@use_docstring(_examples.get_examples_no_header([2]))
class FrameB(_LocalFrame):
    """
    Body frame

    Parameters
    ----------
    point: ECEFvector, GeoPoint or Nvector object
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

    Examples
    --------
    {super}

    See also
    --------
    FrameE, FrameL, FrameN
    """

    _NAMES = ('point', 'yaw', 'pitch', 'roll')

    def __init__(self, point, yaw=0, pitch=0, roll=0, degrees=False):
        self.nvector = point.to_nvector()
        if degrees:
            yaw, pitch, roll = rad(yaw), rad(pitch), rad(roll)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    @property
    def R_EN(self):
        """Rotation matrix to go between E and B frame"""
        R_NB = zyx2R(self.yaw, self.pitch, self.roll)
        n_EB_E = self.nvector.normal
        R_EN = n_E2R_EN(n_EB_E, self.nvector.frame.R_Ee)
        return mdot(R_EN, R_NB)  # rotation matrix

    def _is_equal_to(self, other, rtol=1e-12, atol=1e-14):
        return (allclose(self.yaw, other.yaw, rtol=rtol, atol=atol)
                and allclose(self.pitch, other.pitch, rtol=rtol, atol=atol)
                and allclose(self.roll, other.roll, rtol=rtol, atol=atol)
                and allclose(self.R_EN, other.R_EN, rtol=rtol, atol=atol)
                and self.nvector == other.nvector)


def _check_frames(self, other):
    if not self.frame == other.frame:
        raise ValueError('Frames are unequal')


def _default_frame(frame):
    if frame is None:
        return FrameE()
    return frame


_ODICT = globals()
__doc__ = (__doc__  # @ReservedAssignment
           + _make_summary(dict((n, _ODICT[n]) for n in __all__))
           + 'License\n-------\n'
           + _license.__doc__)


if __name__ == "__main__":
    test_docstrings(__file__)
