"""
Object-oriented interface to geodesic functions
===============================================

"""

# pylint: disable=invalid-name
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from karney import geodesic  # @UnresolvedImport
from numpy.linalg import norm

from nvector import _examples, _license
from nvector._common import _make_summary, test_docstrings, use_docstring
from nvector._typing import (
    Array,
    ArrayLike,
    BoolArray,
    NdArray,
    NpArrayLike,
    format_docstring_types,
)
from nvector.core import (
    _interp_vectors,
    closest_point_on_great_circle,
    course_over_ground,
    cross_track_distance,
    euclidean_distance,
    great_circle_distance,
    intersect,
    lat_lon2n_E,
    n_E2lat_lon,
    n_EA_E_distance_and_azimuth2n_EB_E,
    n_EB_E2p_EB_E,
    on_great_circle_path,
    p_EB_E2n_EB_E,
)
from nvector.rotation import E_rotation, n_E2R_EN, n_E_and_wa2R_EL, zyx2R
from nvector.util import allclose, array_to_list_dict, get_ellipsoid, isclose, mdot, unit

__all__ = [
    "delta_E",
    "delta_L",
    "delta_N",
    "FrameB",
    "FrameE",
    "FrameN",
    "FrameL",
    "GeoPath",
    "GeoPoint",
    "ECEFvector",
    "Nvector",
    "Pvector",
]


@use_docstring(_examples.get_examples_no_header([1]))
def delta_E(
    point_a: Union["Nvector", "GeoPoint", "ECEFvector"],
    point_b: Union["Nvector", "GeoPoint", "ECEFvector"],
) -> "ECEFvector":
    """
    Returns cartesian delta vector from positions A to B decomposed in E.

    Parameters
    ----------
    point_a: Nvector, GeoPoint or ECEFvector
        Position A decomposed in E.
    point_b: Nvector, GeoPoint or ECEFvector
        Position B decomposed in E.

    Returns
    -------
    p_AB_E: ECEFvector
        Cartesian position vector(s) from A to B, decomposed in E.

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
    nvector.core.n_EA_E_and_p_AB_E2n_EB_E,
    nvector.core.p_EB_E2n_EB_E,
    nvector.core.n_EB_E2p_EB_E
    """
    # Function 1. in Section 5.4 in Gade (2010):
    p_EA_E = point_a.to_ecef_vector()
    p_EB_E = point_b.to_ecef_vector()
    p_AB_E = p_EB_E - p_EA_E
    return p_AB_E


def _base_angle(angle_rad: ArrayLike) -> NdArray:
    r"""Returns angle so it is between $-\pi$ and $\pi$"""
    angle_rad_arr = np.asarray(angle_rad)
    return np.mod(angle_rad_arr + np.pi, 2 * np.pi) - np.pi


def delta_N(
    point_a: Union["Nvector", "GeoPoint", "ECEFvector"],
    point_b: Union["Nvector", "GeoPoint", "ECEFvector"],
) -> "Pvector":
    """Returns cartesian delta vector from positions A to B decomposed in N.

    Parameters
    ----------
    point_a: Nvector, GeoPoint or ECEFvector
        Position A decomposed in E.
    point_b: Nvector, GeoPoint or ECEFvector
        Position B decomposed in E.

    Returns
    -------
    p_AB_N: Pvector
        Delta vector from positions A to B, decomposed in N.

    See also
    --------
    delta_E, delta_L
    """
    # p_ab_E = delta_E(point_a, point_b)
    # p_ab_N = p_ab_E.change_frame(....)
    return delta_E(point_a, point_b).change_frame(FrameN.from_point(point_a))


def _delta(
    self: Union["Nvector", "GeoPoint", "ECEFvector"],
    other: Union["Nvector", "GeoPoint", "ECEFvector"],
) -> "Pvector":
    """
    Returns cartesian delta vector from current position to the other decomposed in N.

    Parameters
    ----------
    other: Nvector, GeoPoint or ECEFvector
        Other position decomposed in E.

    Returns
    -------
    p_AB_N: Pvector
        Delta vector from current position (A) to the other position (B), decomposed in N.
    """
    return delta_N(self, other)


def delta_L(
    point_a: Union["Nvector", "GeoPoint", "ECEFvector"],
    point_b: Union["Nvector", "GeoPoint", "ECEFvector"],
    wander_azimuth: Union[int, float] = 0,
) -> "Pvector":
    """Returns cartesian delta vector from positions A to B decomposed in L.

    Parameters
    ----------
    point_a: Nvector, GeoPoint or ECEFvector
        Position A decomposed in E.
    point_b: Nvector, GeoPoint or ECEFvector
        Position B decomposed in E.
    wander_azimuth: real scalar
        Angle [rad] between the x-axis of L and the north direction.

    Returns
    -------
    p_AB_L: Pvector
        Cartesian delta vector from positions A to B decomposed in L.

    See also
    --------
    delta_E, delta_N
    """
    local_frame = FrameL.from_point(point_a, wander_azimuth=wander_azimuth)
    # p_ab_E = delta_E(point_a, point_b)
    # p_ab_L = p_ab_E.change_frame(....)
    return delta_E(point_a, point_b).change_frame(local_frame)


class _Common:
    """Class that defines the common methods for geodetic vector-like and frame-like classes"""

    _NAMES: tuple[str, ...] = ()

    def __repr__(self) -> str:
        cname = self.__class__.__name__
        fmt = ", "
        names = self._NAMES if self._NAMES else list(self.__dict__)
        dict_params = array_to_list_dict(self.__dict__.copy())
        pars = [
            f"{name}={dict_params[name]!r}"  # type: ignore
            for name in names
            if not name.startswith("_")
        ]
        params = fmt.join(pars)
        return f"{cname}({params})"

    def __str__(self) -> str:
        """display a nice short string representation of object."""
        return self._mystr(pretty=True)

    def _mystr(self, pretty: bool = True) -> str:
        """display a nice short string representation of object."""

        def strfun(cls: Any) -> str:
            if isinstance(cls, _Common):
                return cls._mystr(pretty)
            return str(cls)

        def _get_short_arg(name: str, val: Any) -> str:
            fmt = "{}={}"
            if isinstance(val, list) and val and not isinstance(val[0], str):
                val_txts = [strfun(v) for v in val]
                n = sum(map(len, val_txts))
                if pretty and n > 80:
                    val_txts = [arg.replace("\n", "\n    ") for arg in val_txts]
                    val_str = "[\n    {}]".format(",\n    ".join(val_txts))
                else:
                    val_str = "[{}]".format(", ".join(val_txts))
                val = val_str
            elif isinstance(val, str):
                fmt = "{}='{}'"
            return fmt.format(name, strfun(val))

        return self._get_str(_get_short_arg, pretty)

    def _get_str(self, get_arg: Callable[[str, Any], Optional[str]], pretty: bool = False) -> str:
        class_name = self.__class__.__name__
        args: List[str] = []

        names = self._NAMES if self._NAMES else list(self.__dict__)

        n = len(class_name) + 2

        for name in names:
            if not name.startswith("_"):
                val = getattr(self, name)
                val_txt = get_arg(name, val)
                if val_txt is not None:
                    n += len(val_txt) + 2
                    args.append(val_txt)
        if pretty and n > 80:
            args = [arg.replace("\n", "\n    ") for arg in args]
            argstxt = ",\n    ".join(args)
            argstxt = "\n    " + argstxt
            return f"{class_name}({argstxt})"
        return f"{class_name}({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        try:
            return self is other or self._is_equal_to(other, rtol=1e-12, atol=1e-14)
        except AttributeError:  #  , NotImplementedError, TypeError):
            return False

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return not equal  # if equal is not NotImplemented else NotImplemented

    def _is_equal_to(self, other: Any, rtol: float, atol: float) -> bool:
        """Compares another object attributes of the same type"""
        raise NotImplementedError


class GeoPoint(_Common):
    """
    Geographical position(s) given as latitude(s), longitude(s), depth(s) in frame E.

    Attributes
    ----------
    latitude, longitude: ndarray
        Geodetic latitude(s) and longitude(s) given in [rad]
    z: ndarray
        Depth(s) [m] relative to the ellipsoid (depth = -height)
    frame: FrameE
        Reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    Notes
    -----
    Please note that latitude, longitude and z are broadcasted together in the __init__ function.
    If either one of them is a vector the GeoPoint instance then will represents
    multiple positions.

    Examples
    --------
    Solve geodesic problems.

    The following illustrates its use

    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name="WGS84")
    >>> point_a = wgs84.GeoPointFromDegrees(-41.32, 174.81)
    >>> point_b = wgs84.GeoPointFromDegrees(40.96, -5.50)

    >>> print(point_a)
    GeoPoint(
        latitude=-0.721170046924057,
        longitude=3.0510100654112877,
        z=0,
        frame=FrameE(a=6378137.0, f=0.0033528106647474805, name='WGS84', axes='e'))

    The geodesic inverse problem

    >>> s12, az1, az2 = point_a.distance_and_azimuth(point_b, degrees=True)
    >>> "s12 = {:5.2f}, az1 = {:5.2f}, az2 = {:5.2f}".format(s12, az1, az2)
    's12 = 19959679.27, az1 = 161.07, az2 = 18.83'

    The geodesic direct problem

    >>> point_a = wgs84.GeoPointFromDegrees(40.6, -73.8)
    >>> az1, distance = 45, 10000e3
    >>> point_b, az2 = point_a.displace(distance, az1, degrees=True)
    >>> lat2, lon2 = point_b.latitude_deg, point_b.longitude_deg
    >>> msg = "lat2 = {:5.2f}, lon2 = {:5.2f}, az2 = {:5.2f}"
    >>> msg.format(lat2, lon2, az2)
    'lat2 = 32.64, lon2 = 49.01, az2 = 140.37'

    """

    _NAMES = ("latitude", "longitude", "z", "frame")
    latitude: NdArray
    longitude: NdArray
    z: NdArray
    frame: "FrameE"

    def __init__(
        self,
        latitude: ArrayLike,
        longitude: ArrayLike,
        z: ArrayLike = 0,
        frame: Optional["FrameE"] = None,
        degrees: bool = False,
    ) -> None:
        """
        Initialize geographical position given as latitude, longitude, depth in frame E.

        Parameters
        ----------
        latitude : {array_like}
            Geodetic latitude(s) [deg or rad] (scalar or vector)
        longitude : {array_like}
            Geodetic longitude(s) [deg or rad] (scalar or vector)
        z: {array_like}
            Depth(s) [m] relative to the ellipsoid. (depth = -height) (scalar or vector)
        frame: FrameE
            Reference ellipsoid. The default ellipsoid model used is WGS84, but
            other ellipsoids/spheres might be specified.
        degrees: bool
            True if input are given in degrees otherwise radians are assumed.
        """

        if degrees:
            latitude, longitude = np.deg2rad(latitude), np.deg2rad(longitude)
        self.latitude, self.longitude, self.z = np.broadcast_arrays(
            np.asarray(latitude), np.asarray(longitude), np.asarray(z)
        )
        self.frame = _default_frame(frame)

    @classmethod
    @format_docstring_types
    def from_degrees(
        cls,
        latitude: ArrayLike,
        longitude: ArrayLike,
        z: ArrayLike = 0,
        frame: Optional["FrameE"] = None,
    ) -> "GeoPoint":
        """
        Returns GeoPoint from latitude [deg], longitude [deg], depth in frame E.

        Parameters
        ----------
        latitude : {array_like}
            Geodetic latitude(s) [deg] (scalar or vector)
        longitude : {array_like}
            Geodetic longitude(s) [deg] (scalar or vector)
        z: {array_like}
            Depth(s) [m] relative to the ellipsoid. (depth = -height) (scalar or vector)
        frame: FrameE
            Reference ellipsoid. The default ellipsoid model used is WGS84, but
            other ellipsoids/spheres might be specified.
        """
        return cls(latitude, longitude, z, frame, degrees=True)

    def _is_equal_to(self, other: Any, rtol: float = 1e-12, atol: float = 1e-14) -> bool:
        def diff(angle1: NdArray, angle2: NdArray) -> NdArray:
            pi2 = 2 * np.pi
            delta = (np.asarray(angle1) - np.asarray(angle2)) % pi2
            return np.where(delta > np.pi, pi2 - delta, delta)

        delta_lat = diff(self.latitude, other.latitude)
        delta_lon = diff(self.longitude, other.longitude)
        return bool(
            allclose(delta_lat, 0, rtol=rtol, atol=atol)
            and allclose(delta_lon, 0, rtol=rtol, atol=atol)
            and allclose(self.z, other.z, rtol=rtol, atol=atol)
            and self.frame == other.frame
        )

    @property
    def latlon_deg(
        self,
    ) -> tuple[NdArray, NdArray, NdArray]:
        """Latitude [deg], longitude [deg] and depth [m]."""
        return self.latitude_deg, self.longitude_deg, self.z

    @property
    def latlon(
        self,
    ) -> tuple[NdArray, NdArray, NdArray]:
        """Latitude [rad], longitude [rad], and depth [m]."""
        return self.latitude, self.longitude, self.z

    @property
    def latitude_deg(self) -> NdArray:
        """Latitude in degrees."""
        return np.rad2deg(self.latitude)

    @property
    def longitude_deg(self) -> NdArray:
        """Longitude in degrees."""
        return np.rad2deg(self.longitude)

    @property
    def scalar(self) -> bool:
        """True if the position is a scalar point"""
        return np.ndim(self.z) == 0 and np.size(self.latitude) == 1 and np.size(self.longitude) == 1

    def to_ecef_vector(self) -> "ECEFvector":
        """Returns position(s) as ECEFvector object."""
        return self.to_nvector().to_ecef_vector()

    def to_geo_point(self) -> "GeoPoint":
        """Returns position(s) as GeoPoint object, in this case, itself."""
        return self

    def to_nvector(self) -> "Nvector":
        """Returns position(s) as Nvector object."""
        n_vector = lat_lon2n_E(self.latitude, self.longitude, self.frame.R_Ee)
        return Nvector(n_vector, self.z, self.frame)

    delta_to = _delta

    def _displace_great_circle(
        self,
        distance: ArrayLike,
        azimuth: ArrayLike,
        degrees: bool,
    ) -> tuple["GeoPoint", NdArray]:
        """Returns the great circle solution using the nvector method."""
        n_a = self.to_nvector()
        e_a = n_a.to_ecef_vector()
        radius = e_a.length
        distance_rad = np.asarray(distance) / np.asarray(radius)
        azimuth_rad = np.asarray(azimuth) if not degrees else np.deg2rad(azimuth)
        normal_b = n_EA_E_distance_and_azimuth2n_EB_E(n_a.normal, distance_rad, azimuth_rad)
        point_b = Nvector(normal_b, self.z, self.frame).to_geo_point()
        azimuth_b = _base_angle(delta_N(point_b, e_a).azimuth - np.pi)
        if degrees:
            return point_b, np.rad2deg(azimuth_b)
        return point_b, azimuth_b

    @format_docstring_types
    def displace(
        self,
        distance: ArrayLike,
        azimuth: ArrayLike,
        long_unroll: bool = False,
        degrees: bool = False,
        method: str = "ellipsoid",
    ) -> tuple["GeoPoint", NpArrayLike]:
        """
        Returns position b computed from current position, distance and azimuth.

        Parameters
        ----------
        distance: {array_like}
            Ellipsoidal or great circle distance(s) [m] between positions A and B.
        azimuth: {array_like}
            Azimuth(s) [rad or deg] of line(s) at position A.
        long_unroll: bool
            Controls the treatment of longitude when method=="ellipsoid".
            See distance_and_azimuth method for details.
        degrees: bool
            azimuths are given in degrees if True otherwise in radians.
        method: str
            Either "greatcircle" or "ellipsoid", defining the path where to find position B.

        Returns
        -------
        point_b:  GeoPoint
            B position(s).
        azimuth_b: float64 or ndarray
            Azimuth(s) [rad or deg] of line(s) at position(s) B.

        Notes
        -----
        The `karney.geodesic.reckon <https://pypi.python.org/pypi/karney>`_
        function is used When the method is "ellipsoid".
        Keep :math:`|f| <= 1/50` for full double precision accuracy in this case.
        See :cite:`Karney2013Algorithms` for a description of the method.
        """
        if method.lower().startswith("e"):
            return self._displace_ellipsoid(distance, azimuth, long_unroll, degrees)
        return self._displace_great_circle(distance, azimuth, degrees)

    def _displace_ellipsoid(
        self,
        distance: ArrayLike,
        azimuth: ArrayLike,
        long_unroll: bool = False,
        degrees: bool = False,
    ) -> tuple["GeoPoint", NpArrayLike]:
        """Returns the exact ellipsoidal solution using the method of Karney.

        Parameters
        ----------
        distance : {array_like}
            Real scalars or vectors of length n ellipsoidal or great circle distance [m]
            between position A and B.
        azimuth : {array_like}
            Real scalars or vectors of length n azimuth [rad or deg] of line at position A.
        long_unroll : bool
            Controls the treatment of longitude when method=="ellipsoid".
            See distance_and_azimuth method for details.
        degrees : bool
            azimuths are given in degrees if True otherwise in radians.

        Returns
        -------
        point_b:  GeoPoint
            B position(s).
        azimuth_b: {np_array_like}
            Azimuth(s) [rad or deg] of line(s) at position(s) B.
        """
        frame = self.frame
        z = self.z
        azimuth_deg = np.asarray(azimuth) if degrees else np.rad2deg(azimuth)
        lat_a_deg, lon_a_deg = self.latitude_deg, self.longitude_deg
        lat_b_deg, lon_b_deg, azimuth_b_deg = frame.direct(
            lat_a_deg, lon_a_deg, azimuth_deg, distance, z=z, long_unroll=long_unroll, degrees=True
        )
        point_b = frame.GeoPointFromDegrees(latitude=lat_b_deg, longitude=lon_b_deg, z=z)
        if not degrees:
            return point_b, np.deg2rad(azimuth_b_deg)
        return point_b, azimuth_b_deg

    def distance_and_azimuth(
        self,
        point: Union["GeoPoint", "Nvector", "ECEFvector", "Pvector"],
        degrees: bool = False,
        method: str = "ellipsoid",
    ) -> tuple[NpArrayLike, NpArrayLike, NpArrayLike]:
        """
        Returns ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        point: GeoPoint, Nvector, ECEFvector or Pvector
            Geographical position(s) B.
        degrees: bool
            Azimuths are returned in degrees if True otherwise in radians.
        method: str
            Either "greatcircle" or "ellipsoid" defining the path distance calculated.

        Returns
        -------
        s_ab: float64 or ndarray
            Ellipsoidal distance(s) [m] between A and B position(s) at their average height.
        azimuth_a, azimuth_b: float64 or ndarray
            Direction(s) [rad or deg] of line(s) at position A and B relative to
            North, respectively.

        Notes
        -----
        The `karney.geodesic.distance <https://pypi.python.org/pypi/karney>`_
        function is used When the method is "ellipsoid".
        Keep :math:`|f| <= 1/50` for full double precision accuracy in this case.
        See :cite:`Karney2013Algorithms` for a description of the method.

        Examples
        --------
        >>> import nvector as nv
        >>> point1 = nv.GeoPoint(0, 0)
        >>> point2 = nv.GeoPoint.from_degrees(0.5, 179.5)
        >>> s_12, az1, azi2 = point1.distance_and_azimuth(point2)
        >>> bool(nv.allclose(s_12, 19936288.579))
        True

        """
        _check_frames(self, point)
        if method.lower().startswith("e"):
            return self._distance_and_azimuth_ellipsoid(point, degrees)
        return self._distance_and_azimuth_greatcircle(point, degrees)

    def _distance_and_azimuth_greatcircle(
        self, point: Union["GeoPoint", "Nvector", "ECEFvector", "Pvector"], degrees: bool
    ) -> tuple[NpArrayLike, NpArrayLike, NpArrayLike]:
        """
        Returns great circle distance between positions as well as the direction.

        Parameters
        ----------
        point : GeoPoint, Nvector, ECEFvector or Pvector
            Other geographical position(s).
        degrees : bool
            Azimuths are returned in degrees if True otherwise in radians.

        Returns
        -------
        tuple[{np_array_like}, {np_array_like}, {np_array_like}]
        """
        n_a = self.to_nvector()
        n_b = point.to_nvector()
        e_a = n_a.to_ecef_vector()
        e_b = n_b.to_ecef_vector()
        radius = 0.5 * (e_a.length + e_b.length)
        distance = great_circle_distance(n_a.normal, n_b.normal, radius)
        azimuth_a = delta_N(e_a, e_b).azimuth
        azimuth_b = _base_angle(delta_N(e_b, e_a).azimuth - np.pi)

        if degrees:
            azimuth_a, azimuth_b = np.rad2deg(azimuth_a), np.rad2deg(azimuth_b)

        if np.ndim(radius) == 0:
            return np.asarray(distance)[0], azimuth_a, azimuth_b  # scalar track distance
        return distance, azimuth_a, azimuth_b

    def _distance_and_azimuth_ellipsoid(
        self, point: Union["GeoPoint", "Nvector", "ECEFvector", "Pvector"], degrees: bool
    ) -> tuple[NpArrayLike, NpArrayLike, NpArrayLike]:
        """
        Returns ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        point : GeoPoint, Nvector, ECEFvector or Pvector
            Other geographical position(s).
        degrees : bool
            Azimuths are returned in degrees if True otherwise in radians.

        Returns
        -------
        tuple[{np_array_like}, {np_array_like}, {np_array_like}]

        """
        gpoint = point.to_geo_point()
        lat_a, lon_a = self.latitude, self.longitude
        lat_b, lon_b = gpoint.latitude, gpoint.longitude
        z = 0.5 * (self.z + gpoint.z)  # Average depth

        if degrees:
            deg = np.rad2deg
            lat_a, lon_a, lat_b, lon_b = deg(lat_a), deg(lon_a), deg(lat_b), deg(lon_b)

        return self.frame.inverse(lat_a, lon_a, lat_b, lon_b, z, degrees)


_GeoPoint = GeoPoint  # Trick to make typehinting work with mypy and FrameE


class Nvector(_Common):
    """
    Geographical position(s) given as n-vector(s) and depth(s) in frame E

    Attributes
    ----------
    normal: ndarray
        Normal vector(s) [no unit] decomposed in E.
    z: ndarray
        Depth(s) [m] relative to the ellipsoid. (depth = -height)
    frame: FrameE
        Reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.

    Notes
    -----
    The position of B (typically body) relative to E (typically Earth) is
    given into this function as n-vector, n_EB_E and a depth, z relative to the
    ellipsiod.

    Examples
    --------

    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name="WGS84")
    >>> point_a = wgs84.GeoPointFromDegrees(-41.32, 174.81)
    >>> point_b = wgs84.GeoPointFromDegrees(40.96, -5.50)
    >>> nv_a = point_a.to_nvector()
    >>> print(nv_a)
    Nvector(
        normal=[[-0.74795462]
                [ 0.06793758]
                [-0.66026387]],
        z=[0],
        frame=FrameE(a=6378137.0, f=0.0033528106647474805, name='WGS84', axes='e'))

    See also
    --------
    GeoPoint, ECEFvector, Pvector
    """

    _NAMES = ("normal", "z", "frame")
    normal: NdArray
    z: NdArray
    frame: "FrameE"

    def __init__(
        self, normal: ArrayLike, z: ArrayLike = 0, frame: Optional["FrameE"] = None
    ) -> None:
        """
        Initialize geographical position(s) given as n-vector(s) and depth(s) in frame E

        Parameters
        ----------
        normal: 3 x n array
            n-vector(s) [no unit] decomposed in E.
        z: real scalar or vector of length n.
            Depth(s) [m]  relative to the ellipsoid (depth = -height)
        frame: FrameE
            Reference ellipsoid. The default ellipsoid model used is WGS84, but
            other ellipsoids/spheres might be specified.
        """
        normal_arr = np.asarray(normal)
        z_arr = np.asarray(z)
        n = max(normal_arr.shape[1], z_arr.size)
        self.normal = np.broadcast_to(normal_arr, (3, n))
        self.z = np.broadcast_to(z_arr, n)
        self.frame = _default_frame(frame)

    @format_docstring_types
    def interpolate(
        self,
        t_i: ArrayLike,
        t: ArrayLike,
        kind: Union[int, str] = "linear",
        window_length: int = 0,
        polyorder: int = 2,
        mode: str = "interp",
        cval: Union[int, float] = 0.0,
    ) -> "Nvector":
        """
        Returns interpolated values from nvector data.

        Parameters
        ----------
        t_i : {array_like}
            Real vector of length m. Vector of interpolation times.
        t : {array}
            Real vector of length n. Vector of times.
        kind: str or int
            Specifies the kind of interpolation as a string
            ("linear", "nearest", "zero", "slinear", "quadratic", "cubic"
            where "zero", "slinear", "quadratic" and "cubic" refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is "linear".
        window_length: int
            The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
            Must be positive odd integer or zero. Default window_length=0, i.e. no smoothing.
        polyorder: int
            The order of the polynomial used to fit the samples.
            polyorder must be less than window_length.
        mode: str
            Accepted values are "mirror", "constant", "nearest", "wrap" or "interp".
            Determines the type of extension to use for the padded signal to
            which the filter is applied.  When mode is "constant", the padding
            value is given by cval.
            When the "interp" mode is selected (the default), no extension
            is used.  Instead, a degree polyorder polynomial is fit to the
            last window_length values of the edges, and this polynomial is
            used to evaluate the last window_length // 2 output values.
        cval: int or float
            Value to fill past the edges of the input if mode is "constant".
            Default is 0.0.

        Returns
        -------
        Nvector:
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
        >>> nvectors = nv.GeoPoint.from_degrees(lat, lon).to_nvector()
        >>> t = np.arange(10)
        >>> t_i = np.linspace(0, t[-1], 100)
        >>> nvectors_i = nvectors.interpolate(t_i, t, kind="cubic")
        >>> lati, loni, zi = nvectors_i.to_geo_point().latlon_deg
        >>> h = plt.plot(lon, lat, "o", loni, lati, "-")
        >>> plt.show() # doctest: +SKIP
        >>> plt.close()
        """
        vectors = np.vstack((self.normal, self.z))
        vectors_i = _interp_vectors(t_i, t, vectors, kind, window_length, polyorder, mode, cval)
        normal = unit(vectors_i[:3], norm_zero_vector=np.nan)
        return Nvector(normal, z=vectors_i[3], frame=self.frame)

    def to_ecef_vector(self) -> "ECEFvector":
        """Returns position(s) as ECEFvector object."""
        frame = self.frame
        a, f, R_Ee = frame.a, frame.f, frame.R_Ee
        pvector = n_EB_E2p_EB_E(self.normal, depth=self.z, a=a, f=f, R_Ee=R_Ee)
        return ECEFvector(pvector, self.frame, scalar=self.scalar)

    @property
    def scalar(self) -> bool:
        """True if the position is a scalar point"""
        return np.size(self.z) == 1 and self.normal.shape[1] == 1

    def to_geo_point(self) -> GeoPoint:
        """Returns position(s) as GeoPoint object."""
        latitude, longitude = n_E2lat_lon(self.normal, R_Ee=self.frame.R_Ee)

        if self.scalar:
            return GeoPoint(latitude[0], longitude[0], self.z[0], self.frame)  # Scalar geo_point
        return GeoPoint(latitude, longitude, self.z, self.frame)

    def to_nvector(self) -> "Nvector":
        """Returns position(s) as Nvector object, in this case, itself."""
        return self

    delta_to = _delta

    def unit(self) -> None:
        """Normalizes self to unit vector(s)"""
        self.normal = unit(self.normal)

    @format_docstring_types
    def course_over_ground(self, **options: Any) -> NpArrayLike:
        """Returns course over ground in radians from nvector positions

        Parameters
        ----------
        **options : dict
            Optional keyword arguments to apply a Savitzky-Golay smoothing filter window.
            No smoothing is applied by default.
            Valid keyword arguments are:

            window_length: int
                The length of the Savitzky-Golay filter window (i.e., the number of coefficients).
                Positive odd integer or zero. Default window_length=0, i.e. no smoothing.
            polyorder: int
                The order of the polynomial used to fit the samples.
                The value must be less than window_length. Default is 2.
            mode: str
                Valid options are: "mirror", "constant", "nearest", "wrap" or "interp".
                Determines the type of extension to use for the padded signal to
                which the filter is applied.  When mode is "constant", the padding
                value is given by cval. When the "nearest" mode is selected (the default)
                the extension contains the nearest input value.
                When the "interp" mode is selected, no extension
                is used.  Instead, a degree polyorder polynomial is fit to the
                last window_length values of the edges, and this polynomial is
                used to evaluate the last window_length // 2 output values.
                Default "nearest".
            cval: int or float
                Value to fill past the edges of the input if mode is "constant".
                Default is 0.0.

        Returns
        -------
        cog: {np_array_like}
            Angle(s) in radians clockwise from True North to the direction towards
            which the vehicle travels. If n<2 NaN is returned.

        Notes
        -----
        Please be aware that this method requires the vehicle positions to be very smooth!
        If they are not you should probably smooth it by a window_length corresponding
        to a few seconds or so. The smoothing filter is only applied if the optional keyword
        `window_length` value is an integer and >0. Also note that at least n>=2 points are needed
        to obtain meaningful results.

        See https://www.navlab.net/Publications/The_Seven_Ways_to_Find_Heading.pdf
        for an overview of methods to find accurate headings.

        Examples
        --------

        >>> import matplotlib.pyplot as plt
        >>> import nvector as nv
        >>> points = nv.GeoPoint.from_degrees((59.381509, 59.387647),(10.496590, 10.494713))
        >>> nvec = points.to_nvector()
        >>> COG_rad = nvec.course_over_ground()
        >>> dx, dy = np.sin(COG_rad[0]), np.cos(COG_rad[0])
        >>> COG = nv.deg(COG_rad[0])
        >>> p_AB_N = nv.n_EA_E_and_n_EB_E2p_AB_N(nvec.normal[:, :1], nvec.normal[:, 1:]).ravel()
        >>> ax = plt.figure().gca()
        >>> _ = ax.plot(0, 0, "bo", label="A")
        >>> _ = ax.arrow(0,0, dx*300, dy*300, head_width=20)
        >>> _ = ax.plot(p_AB_N[1], p_AB_N[0], "go", label="B")
        >>> _ = ax.set_title("COG=%2.1f degrees" % COG)
        >>> _ = ax.set_xlabel("East [m]")
        >>> _ = ax.set_ylabel("North [m]")
        >>> _ = ax.set_xlim(-500, 200)
        >>> _ = ax.set_aspect("equal", adjustable="box")
        >>> _ = ax.legend()
        >>> plt.show() # doctest: +SKIP
        >>> plt.close()

        See also
        --------
        nvector.core.course_over_ground
        """
        frame = self.frame
        return course_over_ground(self.normal, a=frame.a, f=frame.f, R_Ee=frame.R_Ee, **options)

    def mean(self) -> "Nvector":
        """Returns the mean position of the n-vectors."""
        average_nvector = unit(np.sum(self.normal, axis=1, keepdims=True))
        return self.frame.Nvector(average_nvector, z=np.mean(self.z))

    def _is_equal_to(self, other: Any, rtol: float = 1e-12, atol: float = 1e-14) -> bool:
        return bool(
            allclose(self.normal, other.normal, rtol=rtol, atol=atol)
            and allclose(self.z, other.z, rtol=rtol, atol=atol)
            and self.frame == other.frame
        )

    def __add__(self, other: "Nvector") -> "Nvector":
        _check_frames(self, other)
        return self.frame.Nvector(self.normal + other.normal, self.z + other.z)

    def __sub__(self, other: "Nvector") -> "Nvector":
        _check_frames(self, other)
        return self.frame.Nvector(self.normal - other.normal, self.z - other.z)

    def __neg__(self) -> "Nvector":
        return self.frame.Nvector(-self.normal, -self.z)

    def __mul__(self, scalar: Any) -> "Nvector":
        """Elementwise multiplication"""

        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal * scalar, self.z * scalar)
        return NotImplemented  # "Only scalar multiplication is implemented"

    def __truediv__(self, scalar: Any) -> "Nvector":
        """Elementwise division"""
        if not isinstance(scalar, Nvector):
            return self.frame.Nvector(self.normal / scalar, self.z / scalar)
        return NotImplemented  # "Only scalar division is implemented"

    __radd__ = __add__
    __rmul__ = __mul__


class _Pvector(_Common):
    """
    Geographical position(s) given as cartesian position vector(s) in a frame.

    Attributes
    ----------
    pvector : ndarray
        3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
    frame : FrameN, FrameB or FrameL
        Local frame
    scalar : bool
        True if p-vector represents a scalar position, i.e. n = 1.
    """

    _NAMES: Tuple[str, ...] = ("pvector", "frame", "scalar")
    pvector: NdArray
    """Position array-like, must be shape (3, n, m, ...) with n>0"""
    frame: Union["FrameE", "FrameN", "FrameB", "FrameL", "_LocalFrameBase"]
    scalar: bool

    def __init__(
        self,
        pvector: Array,
        frame: Union["FrameN", "FrameB", "FrameL", "_LocalFrameBase"],
        scalar: Optional[bool] = None,
    ) -> None:
        """
        Initialize geographical position(s) given as cartesian position vector(s) in a frame.

        Parameters
        ----------
        pvector : list, tuple or ndarray
            3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
        frame : FrameN, FrameB or FrameL
            Local frame
        scalar : bool
            True if p-vector represents a scalar position.
            If None, then determined by shape of pvector
        """
        pvector_arr = np.asarray(pvector)
        if scalar is None:
            scalar = pvector_arr.shape[1] == 1
        self.pvector = pvector_arr
        self.frame = frame
        self.scalar = scalar

    delta_to = _delta

    def _is_equal_to(self, other: Any, rtol: float = 1e-12, atol: float = 1e-14) -> bool:
        return bool(
            allclose(self.pvector, other.pvector, rtol=rtol, atol=atol)
            and self.frame == other.frame
        )

    @property
    def length(self) -> NpArrayLike:
        """Length of the pvector."""
        lengths = norm(self.pvector, axis=0)
        return lengths[0] if self.scalar else lengths

    @property
    def azimuth_deg(self) -> NpArrayLike:
        """Azimuth in degree clockwise relative to the x-axis."""
        return np.rad2deg(self.azimuth)

    @property
    def azimuth(self) -> NpArrayLike:
        """Azimuth in radian clockwise relative to the x-axis."""
        p_AB_N = self.pvector
        az = np.arctan2(p_AB_N[1], p_AB_N[0])
        return az[0] if self.scalar else az

    @property
    def elevation_deg(self) -> NpArrayLike:
        """Elevation in degree relative to the xy-plane. (Positive downwards in a NED frame)"""
        return np.rad2deg(self.elevation)

    @property
    def elevation(self) -> NpArrayLike:
        """Elevation in radian relative to the xy-plane. (Positive downwards in a NED frame)"""
        z = self.pvector[2]
        length = self.length
        el = np.arcsin(z / length)
        return el[0] if self.scalar else el


class Pvector(_Pvector):
    """
    Geographical position(s) given as cartesian position vector(s) in a frame.

    Attributes
    ----------
    pvector : ndarray
        3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
    frame : FrameN, FrameB or FrameL
        Local frame
    scalar : bool
        True if p-vector represents a scalar position, i.e. n = 1.
    """

    _NAMES: Tuple[str, ...] = ("pvector", "frame", "scalar")
    pvector: NdArray
    """Position array-like, must be shape (3, n, m, ...) with n>0"""
    frame: Union["FrameN", "FrameB", "FrameL", "_LocalFrameBase"]
    scalar: bool

    def __init__(
        self,
        pvector: Array,
        frame: Union["FrameN", "FrameB", "FrameL", "_LocalFrameBase"],
        scalar: Optional[bool] = None,
    ) -> None:
        """
        Initialize geographical position(s) given as cartesian position vector(s) in a frame.

        Parameters
        ----------
        pvector : list, tuple or ndarray
            3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
        frame : FrameN, FrameB or FrameL
            Local frame
        scalar : bool
            True if p-vector represents a scalar position.
            If None, then determined by shape of pvector
        """
        pvector_arr = np.asarray(pvector)
        if scalar is None:
            scalar = pvector_arr.shape[1] == 1
        self.pvector = pvector_arr
        self.frame = frame
        self.scalar = scalar

    def to_ecef_vector(self) -> "ECEFvector":
        """Returns position(s) as ECEFvector object."""
        n_frame = self.frame
        p_AB_N = self.pvector
        # alternatively: np.dot(n_frame.R_EN, p_AB_N)
        p_AB_E = mdot(n_frame.R_EN, p_AB_N[:, None, ...]).reshape(3, -1)
        return ECEFvector(p_AB_E, frame=n_frame.nvector.frame, scalar=self.scalar)

    def to_nvector(self) -> Nvector:
        """Returns position(s) as Nvector object."""
        return self.to_ecef_vector().to_nvector()

    def to_geo_point(self) -> GeoPoint:
        """Returns position(s) as GeoPoint object."""
        return self.to_ecef_vector().to_geo_point()


@use_docstring(_examples.get_examples_no_header([3, 4]))
class ECEFvector(_Pvector):  # _Common):
    """
    Geographical position(s) given as cartesian position vector(s) in frame E

    Attributes
    ----------
    pvector: ndarray
        3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
    frame: FrameE
        Reference ellipsoid. The default ellipsoid model used is WGS84, but
        other ellipsoids/spheres might be specified.
    scalar : bool
        True if p-vector represents a scalar position, i.e. n = 1.

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

    _NAMES = ("pvector", "frame", "scalar")
    pvector: NdArray
    """Position array-like, must be shape (3, n, m, ...) with n>0"""
    frame: "FrameE"
    scalar: bool

    def __init__(
        self, pvector: Array, frame: Optional["FrameE"] = None, scalar: Optional[bool] = None
    ):
        """
        Initialize geographical position(s) given as cartesian position vector(s) in a frame.

        Parameters
        ----------
        pvector : list, tuple or ndarray
            3 x n array cartesian position vector(s) [m] from E to B, decomposed in E.
        frame : FrameE
            Local frame
        scalar : bool
            True if p-vector represents a scalar position, i.e. n = 1.
        """
        pvector_arr = np.asarray(pvector)
        if scalar is None:
            scalar = pvector_arr.shape[1] == 1
        self.pvector = pvector_arr
        self.frame = _default_frame(frame)
        self.scalar = scalar

    def change_frame(self, frame: Union["FrameB", "FrameL", "FrameN"]) -> Pvector:
        """
        Converts to Cartesian position vector in another frame

        Parameters
        ----------
        frame : FrameB, FrameN or FrameL
            Local frame M used to convert p_AB_E (position vector from A to B,
            decomposed in E) to a cartesian vector p_AB_M decomposed in M.

        Returns
        -------
        p_AB_M : Pvector
            Position vector from A to B, decomposed in frame M.

        See also
        --------
        n_EB_E2p_EB_E, n_EA_E_and_p_AB_E2n_EB_E, n_EA_E_and_n_EB_E2p_AB_E.
        """
        _check_frames(self, frame.nvector)
        p_AB_E = self.pvector
        p_AB_N = mdot(np.swapaxes(frame.R_EN, 1, 0), p_AB_E[:, None, ...])
        return Pvector(p_AB_N.reshape(3, -1), frame=frame, scalar=self.scalar)

    def to_ecef_vector(self) -> "ECEFvector":
        """Returns position(s) as ECEFvector object, in this case, itself."""
        return self

    def to_geo_point(self) -> GeoPoint:
        """Returns position(s) as GeoPoint object."""
        return self.to_nvector().to_geo_point()

    def to_nvector(self) -> Nvector:
        """Returns position(s) as Nvector object."""
        frame = self.frame
        p_EB_E = self.pvector
        R_Ee = frame.R_Ee
        n_EB_E, depth = p_EB_E2n_EB_E(p_EB_E, a=frame.a, f=frame.f, R_Ee=R_Ee)
        if self.scalar:
            return Nvector(n_EB_E, z=depth[0], frame=frame)
        return Nvector(n_EB_E, z=depth, frame=frame)

    delta_to = _delta

    def __add__(self, other: "ECEFvector") -> "ECEFvector":
        _check_frames(self, other)
        scalar = self.scalar and other.scalar
        return ECEFvector(self.pvector + other.pvector, self.frame, scalar)

    def __sub__(self, other: "ECEFvector") -> "ECEFvector":
        _check_frames(self, other)
        scalar = self.scalar and other.scalar
        return ECEFvector(self.pvector - other.pvector, self.frame, scalar)

    def __neg__(self) -> "ECEFvector":
        return ECEFvector(-self.pvector, self.frame, self.scalar)


@use_docstring(_examples.get_examples_no_header([5, 6, 9, 10]))
class GeoPath(_Common):
    """
    Geographical path between two positions in Frame E

    Attributes
    ----------
    point_a, point_b: Nvector, GeoPoint or ECEFvector
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

    _NAMES = ("point_a", "point_b")
    point_a: Union[Nvector, GeoPoint, ECEFvector]
    point_b: Union[Nvector, GeoPoint, ECEFvector]

    def __init__(
        self,
        point_a: Union[Nvector, GeoPoint, ECEFvector],
        point_b: Union[Nvector, GeoPoint, ECEFvector],
    ) -> None:
        """
        Initialize geographical path between two positions in Frame E

        Parameters
        ----------
        point_a : Nvector, GeoPoint or ECEFvector
            Starting point of path, position A, decomposed in E.
        point_b : Nvector, GeoPoint or ECEFvector
            Ending point of path, position B, decomposed in E.
        """
        self.point_a = point_a
        self.point_b = point_b

    def _is_equal_to(self, other: Any, rtol: float, atol: float) -> bool:
        """Compares another object attributes of the same type"""
        return self.point_a == other.point_a and self.point_b == other.point_b

    def nvectors(self) -> tuple[Nvector, Nvector]:
        """Returns point A and point B as n-vectors"""
        return self.point_a.to_nvector(), self.point_b.to_nvector()

    def geo_points(self) -> tuple[GeoPoint, GeoPoint]:
        """Returns point A and point B as geo-points"""
        return self.point_a.to_geo_point(), self.point_b.to_geo_point()

    def ecef_vectors(self) -> tuple[ECEFvector, ECEFvector]:
        """Returns point A and point B as  ECEF-vectors"""
        return self.point_a.to_ecef_vector(), self.point_b.to_ecef_vector()

    def nvector_normals(self) -> tuple[NdArray, NdArray]:
        """Returns nvector normals for position a and b"""
        nvector_a, nvector_b = self.nvectors()
        return nvector_a.normal, nvector_b.normal

    def _get_average_radius(self) -> NpArrayLike:
        p_E1_E, p_E2_E = self.ecef_vectors()
        return (p_E1_E.length + p_E2_E.length) / 2

    def cross_track_distance(
        self,
        point: Union[Nvector, GeoPoint, ECEFvector],
        method: str = "greatcircle",
        radius: Optional[NpArrayLike] = None,
    ) -> NpArrayLike:
        """
        Returns cross track distance from path to point.

        Parameters
        ----------
        point: GeoPoint, Nvector or ECEFvector
            Position(s) to measure the cross track distance to.
        method: str
            Either "greatcircle" or "euclidean" defining distance calculated.
        radius: real scalar
            Radius of sphere in [m]. Default is the average height of points A and B.


        Returns
        -------
        distance: real scalar or vector
            Distance(s) in [m]

        Notes
        -----
        The result for spherical Earth is returned.
        """
        if radius is None:
            radi = self._get_average_radius()
        else:
            radi = radius
        path = self.nvector_normals()
        n_c = point.to_nvector().normal
        distance = cross_track_distance(path, n_c, method=method, radius=np.asarray(radi))
        if np.ndim(radi) == 0 and np.size(distance) == 1:
            return np.asarray(distance)[0]
        return distance

    def track_distance(
        self, method: str = "greatcircle", radius: Optional[float] = None
    ) -> NpArrayLike:
        """
        Returns the path distance computed at the average height in [m].

        Parameters
        ----------
        method: str
            "greatcircle", "euclidean" or "ellipsoidal" defining distance calculated.
        radius: real scalar
            Radius of sphere. Default is the average height of points A and B
        """
        ellipsod = method.lower().startswith("ex") or method.lower().startswith("el")
        if ellipsod:  # exact or ellipsoidal
            point_a, point_b = self.geo_points()
            s_ab, _, _ = point_a.distance_and_azimuth(point_b, method="ellipsoid")
            return s_ab

        effective_radius = radius if radius is not None else self._get_average_radius()
        normal_a, normal_b = self.nvector_normals()

        euclidean = method.lower().startswith("eu")
        distance_fun = euclidean_distance if euclidean else great_circle_distance
        distance = distance_fun(normal_a, normal_b, np.asarray(effective_radius))

        if np.ndim(effective_radius) == 0:
            return np.asarray(distance)[0]
        return distance

    def intersect(self, path: "GeoPath") -> Nvector:
        """
        Returns the intersection(s) between the great circles of the two paths

        Parameters
        ----------
        path: GeoPath
            Path to intersect

        Returns
        -------
        point: Nvector
            Intersection(s) between the great circles of the two paths

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
        depth = (point_a1.z + point_a2.z + point_b1.z + point_b2.z) / 4.0
        return frame.Nvector(normal_c, z=depth)

    def _on_ellipsoid_path(
        self, point: Union[Nvector, GeoPoint, ECEFvector], rtol: float = 1e-6, atol: float = 1e-8
    ) -> BoolArray:
        point_a, point_b = self.geo_points()
        point_c = point.to_geo_point()
        z = (point_a.z + point_b.z) * 0.5
        distance_ab, azimuth_ab, _ = point_a.distance_and_azimuth(point_b)
        distance_ac, azimuth_ac, _ = point_a.distance_and_azimuth(point_c)
        return isclose(z, point_c.z, rtol=rtol, atol=atol) & (
            isclose(distance_ac, 0, atol=atol)
            | ((distance_ab >= distance_ac) & isclose(azimuth_ac, azimuth_ab, rtol=rtol, atol=atol))
        )

    def on_great_circle(
        self, point: Union[Nvector, GeoPoint, ECEFvector], atol: float = 1e-8
    ) -> Union[bool, BoolArray]:
        """Returns True if point is on the great circle within a tolerance."""
        distance = np.abs(self.cross_track_distance(point))
        result = isclose(distance, 0, atol=atol)
        if np.ndim(result) == 0:
            return result[()]
        return result

    def _on_great_circle_path(
        self,
        point: Union[Nvector, GeoPoint, ECEFvector],
        # radius: Optional[float] = None,
        rtol: float = 1e-9,
        atol: float = 1e-8,
    ) -> BoolArray:
        # if radius is None:
        radi = self._get_average_radius()
        # else:
        #    radi = radius
        n_a, n_b = self.nvectors()
        path = (n_a.normal, n_b.normal)
        n_c = point.to_nvector()
        same_z = isclose(n_c.z, (n_a.z + n_b.z) * 0.5, rtol=rtol, atol=atol)
        result = on_great_circle_path(path, n_c.normal, np.asarray(radi), atol=atol) & same_z
        if np.ndim(radi) == 0 and result.size == 1:
            return result[0]  # scalar outout
        return result

    def on_path(
        self,
        point: Union[Nvector, GeoPoint, ECEFvector],
        method: str = "greatcircle",
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> BoolArray:
        """
        Returns True if point is on the path between A and B witin a tolerance.

        Parameters
        ----------
        point : Nvector, GeoPoint or ECEFvector
            Point to test
        method: "greatcircle" or "ellipsoid"
            Defines the path.
        rtol : real scalar
            The relative tolerance parameter.
        atol : real scalar
            The absolute tolerance parameter.

        Returns
        -------
        result: Boolean vector
            True if the point is on the path at its average height.

        Notes
        -----
        The result for spherical Earth is returned for method="greatcircle".

        Examples
        --------
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name="WGS84")
        >>> pointA = wgs84.GeoPointFromDegrees(89, 0)
        >>> pointB = wgs84.GeoPointFromDegrees(80, 0)
        >>> path = nv.GeoPath(pointA, pointB)
        >>> pointC = path.interpolate(0.6).to_geo_point()
        >>> bool(path.on_path(pointC))
        True
        >>> bool(path.on_path(pointC, "ellipsoid"))
        True
        >>> pointD = path.interpolate(1.000000001).to_geo_point()
        >>> bool(path.on_path(pointD))
        False
        >>> bool(path.on_path(pointD, "ellipsoid"))
        False
        >>> pointE = wgs84.GeoPointFromDegrees(85, 0.0001)
        >>> bool(path.on_path(pointE))
        False
        >>> pointC = path.interpolate(-2).to_geo_point()
        >>> bool(path.on_path(pointC))
        False
        >>> bool(path.on_great_circle(pointC))
        True
        """
        if method[:2].lower() in {"ex", "el"}:  # exact or ellipsoid
            return self._on_ellipsoid_path(point, rtol=rtol, atol=atol)
        return self._on_great_circle_path(point, rtol=rtol, atol=atol)

    def _closest_point_on_great_circle(
        self, point: Union[Nvector, GeoPoint, ECEFvector]
    ) -> Nvector:
        point_c = point.to_nvector()
        point_a, point_b = self.nvectors()
        path = (point_a.normal, point_b.normal)
        z = (point_a.z + point_b.z) * 0.5
        normal_d = closest_point_on_great_circle(path, point_c.normal)
        return point_c.frame.Nvector(normal_d, z)

    def closest_point_on_great_circle(
        self, point: Union[Nvector, GeoPoint, ECEFvector]
    ) -> GeoPoint:
        """
        Returns closest point on great circle path to the point.

        Parameters
        ----------
        point: GeoPoint, Nvector or ECEFvector
            Point of intersection between paths

        Returns
        -------
        closest_point: GeoPoint
            Closest point on path.

        Notes
        -----
        The result for spherical Earth is returned at the average depth.

        Examples
        --------
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name="WGS84")
        >>> point_a = wgs84.GeoPoint(51., 1., degrees=True)
        >>> point_b = wgs84.GeoPoint(51., 2., degrees=True)
        >>> point_c = wgs84.GeoPoint(51., 2.9, degrees=True)
        >>> path = nv.GeoPath(point_a, point_b)
        >>> point = path.closest_point_on_great_circle(point_c)
        >>> bool(path.on_path(point))
        False
        >>> bool(nv.allclose((point.latitude_deg, point.longitude_deg),
        ...                  (50.99270338, 2.89977984)))
        True

        >>> bool(nv.allclose(GeoPath(point_c, point).track_distance(),  810.76312076))
        True

        """

        point_d = self._closest_point_on_great_circle(point)
        return point_d.to_geo_point()

    def closest_point_on_path(self, point: Union[Nvector, GeoPoint, ECEFvector]) -> GeoPoint:
        """
        Returns closest point on great circle path segment to the point.

        If the point is within the extent of the segment, the point returned is
        on the segment path otherwise, it is the closest endpoint defining the
        path segment.

        Parameters
        ----------
        point: GeoPoint
            Point of intersection between paths

        Returns
        -------
        closest_point: GeoPoint
            Closest point on path segment.

        Examples
        --------
        >>> import nvector as nv
        >>> wgs84 = nv.FrameE(name="WGS84")
        >>> pointA = wgs84.GeoPoint(51., 1., degrees=True)
        >>> pointB = wgs84.GeoPoint(51., 2., degrees=True)
        >>> pointC = wgs84.GeoPoint(51., 1.9, degrees=True)
        >>> path = nv.GeoPath(pointA, pointB)
        >>> point = path.closest_point_on_path(pointC)
        >>> bool(np.allclose((point.latitude_deg, point.longitude_deg),
        ...                  (51.00038411380564, 1.900003311624411)))
        True
        >>> bool(np.allclose(GeoPath(pointC, point).track_distance(),  42.67368351))
        True
        >>> pointD = wgs84.GeoPoint(51.0, 2.1, degrees=True)
        >>> pointE = path.closest_point_on_path(pointD) # 51.0000, 002.0000
        >>> float(pointE.latitude_deg), float(pointE.longitude_deg)
        (51.0, 2.0)
        """
        # TODO: vectorize this
        return self._closest_point_on_path(point)

    def _closest_point_on_path(self, point: Union[Nvector, GeoPoint, ECEFvector]) -> GeoPoint:
        point_c = self._closest_point_on_great_circle(point)
        if self.on_path(point_c):
            return point_c.to_geo_point()
        n0 = point.to_nvector().normal
        n1, n2 = self.nvector_normals()
        radius = self._get_average_radius()
        d1 = great_circle_distance(n1, n0, np.asarray(radius))
        d2 = great_circle_distance(n2, n0, np.asarray(radius))
        if np.all(d1 < d2):
            return self.point_a.to_geo_point()
        return self.point_b.to_geo_point()

    @format_docstring_types
    def interpolate(self, ti: ArrayLike) -> Nvector:
        """
        Returns the interpolated point along the path

        Parameters
        ----------
        ti: {array_like}
            Interpolation time(s) assuming position A and B is at t0=0 and t1=1,
            respectively.

        Returns
        -------
        point: Nvector
            Point of interpolation along path
        """
        point_a, point_b = self.nvectors()
        point_c = point_a + (point_b - point_a) * np.asarray(ti)
        point_c.normal = unit(point_c.normal, norm_zero_vector=np.nan)
        return point_c


class FrameE(_Common):
    """Earth-fixed frame

    Attributes
    ----------
    a: float
        Semi-major axis of the Earth ellipsoid given in [m].
    f: float
        Flattening [no unit] of the Earth ellipsoid. If f==0 then spherical
        Earth with radius a.
    name: str
        Defines the default ellipsoid if not `a` or `f` are specified. Default "WGS84".
        See get_ellipsoid for valid options.
    axes: str
        Either "e" or "E". Defines axes orientation of E frame. Default is axes="e" which means
        that the orientation of the axis is such that:
        z-axis -> North Pole, x-axis -> Latitude=Longitude=0.

    Notes
    -----
    The frame is Earth-fixed (rotates and moves with the Earth) where the
    origin coincides with Earth's centre (geometrical centre of ellipsoid
    model).

    See also
    --------
    FrameN, FrameL, FrameB, nvector.util.get_ellipsoid
    """

    _NAMES = ("a", "f", "name", "axes")
    a: float
    f: float
    name: str
    axes: str

    def __init__(
        self,
        a: Optional[float] = None,
        f: Optional[float] = None,
        name: str = "WGS84",
        axes: str = "e",
    ) -> None:
        if a is None or f is None:
            a, f, _full_name = get_ellipsoid(name)
        self.a = a
        self.f = f
        self.name = name
        self.axes = axes

    @property
    def R_Ee(self) -> NdArray:
        """Rotation matrix R_Ee defining the axes of the coordinate frame E"""
        return E_rotation(self.axes)

    def _is_equal_to(self, other: Any, rtol: float = 1e-12, atol: float = 1e-14) -> bool:
        return (
            allclose(self.a, other.a, rtol=rtol, atol=atol)
            and allclose(self.f, other.f, rtol=rtol, atol=atol)
            and allclose(self.R_Ee, other.R_Ee, rtol=rtol, atol=atol)
        )

    @format_docstring_types
    def inverse(
        self,
        lat_a: ArrayLike,
        lon_a: ArrayLike,
        lat_b: ArrayLike,
        lon_b: ArrayLike,
        z: ArrayLike = 0,
        degrees: bool = False,
    ) -> tuple[NpArrayLike, NpArrayLike, NpArrayLike]:
        """
        Returns ellipsoidal distance between positions as well as the direction.

        Parameters
        ----------
        lat_a : {array_like}
            Scalar or vectors of latitude of position A.
        lon_a : {array_like}
            Scalar or vectors of longitude of position A.
        lat_b : {array_like}
            Scalar or vectors of latitude of position B.
        lon_b : {array_like}
            Scalar or vectors of longitude of position B.
        z : {array_like}
            Scalar or vectors of depth relative to Earth ellipsoid (default = 0)
        degrees : bool
            Angles are given in degrees if True otherwise in radians.

        Returns
        -------
        s_ab: {np_array_like}
            Ellipsoidal distance [m] between position A and B.
        azimuth_a, azimuth_b:  {np_array_like}
            Direction [rad or deg] of line at position A and B relative to
            North, respectively.

        Notes
        -----
        This method is a thin wrapper around the
        `karney.geodesic.distance <https://pypi.python.org/pypi/karney>`_ function,
        which is an implementation of the method described in :cite:`Karney2013Algorithms`.

        Restriction on the parameters:
          * Latitudes must lie between -90 and 90 degrees.
          * Latitudes outside this range will be set to NaNs.
          * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        """
        a1 = self.a - np.asarray(z)
        return geodesic.distance(
            np.asarray(lat_a),
            np.asarray(lon_a),
            np.asarray(lat_b),
            np.asarray(lon_b),
            a1,
            self.f,
            degrees=degrees,
        )

    @format_docstring_types
    def direct(
        self,
        lat_a: ArrayLike,
        lon_a: ArrayLike,
        azimuth: ArrayLike,
        distance: ArrayLike,
        z: ArrayLike = 0,
        long_unroll: bool = False,
        degrees: bool = False,
    ) -> tuple[NpArrayLike, NpArrayLike, NpArrayLike]:
        """
        Returns position B computed from position A, distance and azimuth.

        Parameters
        ----------
        lat_a : {array_like}
            Scalar or length n vector of latitude of position A.
        lon_a : {array_like}
            Scalar or length n vector of longitude of position A.
        azimuth : {array_like}
            Scalar or length n vector azimuth [rad or deg] of line at position A relative to North.
        distance : {array_like}
            Scalar or length n vector ellipsoidal distance [m] between position A and B.
        z : {array_like}
            Scalar or length n vector depth relative to Earth ellipsoid (default = 0).
        long_unroll: bool
            Controls the treatment of longitude. If it is False then the lon_a and lon_b
            are both reduced to the range [-180, 180). If it is True, then lon_a
            is as given in the function call and (lon_b - lon_a) determines how many times
            and in what sense the geodesic has encircled the ellipsoid.
        degrees: bool
            Angles are given in degrees if True otherwise in radians.

        Returns
        -------
        lat_b, lon_b: {np_array_like}
            Latitude(s) and longitude(s) of position B. (Scalar or vector)
        azimuth_b: {np_array_like}
            Azimuth(s) [rad or deg] of line(s) at position B relative to North.

        Notes
        -----
        This method is a thin wrapper around the
        `karney.geodesic.reckon <https://pypi.python.org/pypi/karney>`_ function,
        which is an implementation of the method described in :cite:`Karney2013Algorithms`.

        Restriction on the parameters:
          * Latitudes must lie between -90 and 90 degrees.
          * Latitudes outside this range will be set to NaNs.
          * The flattening f should be between -1/50 and 1/50 inn order to retain full accuracy.

        """
        a1 = self.a - np.asarray(z)
        lat1, lon1, az1, dist, a1_arr = np.broadcast_arrays(
            np.asarray(lat_a), np.asarray(lon_a), np.asarray(azimuth), np.asarray(distance), a1
        )
        return geodesic.reckon(lat1, lon1, dist, az1, a1_arr, self.f, long_unroll, degrees=degrees)

    @format_docstring_types
    def GeoPoint(
        self, latitude: ArrayLike, longitude: ArrayLike, z: ArrayLike = 0, degrees: bool = False
    ) -> _GeoPoint:
        """
        Returns GeoPoint from latitude, longitude, depth in current frame.

        Parameters
        ----------
        latitude: {array_like}
            Geodetic latitude(s)  given in [rad or deg]
        longitude: {array_like}
            Geodetic longitude(s) given in [rad or deg]
        z: {array_like}
            Depth(s) [m] relative to the ellipsoid (depth = -height)
        degrees: bool
            True if input are given in degrees otherwise radians are assumed.
        """
        return GeoPoint(latitude, longitude, z, frame=self, degrees=degrees)

    @format_docstring_types
    def GeoPointFromDegrees(
        self, latitude: ArrayLike, longitude: ArrayLike, z: ArrayLike = 0
    ) -> _GeoPoint:
        """
        Returns GeoPoint from latitude [deg], longitude [deg], depth in current frame

        Parameters
        ----------
        latitude: {array_like}
            Geodetic latitude(s)  given in [rad or deg]
        longitude: {array_like}
            Geodetic longitude(s) given in [rad or deg]
        z: {array_like}
            Depth(s) [m] relative to the ellipsoid (depth = -height)
        """
        return GeoPoint.from_degrees(latitude, longitude, z, frame=self)

    @format_docstring_types
    def Nvector(self, normal: ArrayLike, z: ArrayLike = 0) -> Nvector:
        """
        Returns Nvector from n-vector(s) and depth(s) in current frame.

        Parameters
        ----------
        normal: {array}
            3 x n array of n-vector(s) [no unit] decomposed in E.
        z: {array_like}
            Depth(s) [m]  relative to the ellipsoid (depth = -height)
        """
        return Nvector(normal, z, frame=self)

    @format_docstring_types
    def ECEFvector(self, pvector: Array, scalar: Optional[bool] = None) -> ECEFvector:
        """
        Returns ECEFvector from cartesian position vector(s) in current frame.

        Parameters
        ----------
        pvector :  {array}
            3 x n array of cartesian position vector(s) [m] from E to B, decomposed in E.
        scalar : bool
            True if p-vector represents a scalar position, i.e. n = 1.
        """
        return ECEFvector(pvector, frame=self, scalar=scalar)


class _LocalFrameBase(_Common):
    nvector: Nvector

    @property
    def R_EN(self) -> NdArray:
        raise NotImplementedError

    def Pvector(self, pvector: Array) -> "Pvector":
        """Returns Pvector relative to the local frame.

        Parameters
        ----------
        pvector : {array}
            3 x n array of cartesian position vector(s) [m] from Q to B, decomposed in Q.
            The frame Q can be B, L or N.

        Returns
        -------
        Pvector

        """
        return Pvector(pvector, frame=self)


@use_docstring(_examples.get_examples_no_header([1]))
class FrameN(_LocalFrameBase):
    """
    North-East-Down frame

    Attributes
    ----------
    nvector: Nvector
        Defines the origin of the local frame N. The origin is directly beneath or
        above the vehicle (B), at the surface of ellipsoid model.

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

    _NAMES: Tuple[str, ...] = ("nvector",)
    nvector: Nvector

    def __init__(self, nvector: Nvector) -> None:
        """
        Initialize the origin of the North-East-Down frame.

        Parameters
        ----------
        nvector: Nvector
            Position of the vehicle (B) which also defines the origin of the local
            frame N. The origin is directly beneath or above the vehicle (B), at the
            surface of ellipsoid model.
        """
        self.nvector = Nvector(nvector.normal, z=0, frame=nvector.frame)

    @classmethod
    def from_point(cls, point: Union[ECEFvector, GeoPoint, Nvector]) -> "FrameN":
        """
        Returns FrameN with its origin projected from the point to the surface of ellipsoid model

        Parameters
        ----------
        point: ECEFvector, GeoPoint or Nvector
            Position of the vehicle (B) which also defines the origin of the local
            frame N. The origin is directly beneath or above the vehicle (B), at the
            surface of ellipsoid model.
        """
        return cls(point.to_nvector())

    @property
    def R_EN(self) -> NdArray:
        """Rotation matrix to go between E and N frame"""
        nvector = self.nvector
        return n_E2R_EN(nvector.normal, nvector.frame.R_Ee)

    def _is_equal_to(self, other: Any, rtol: float = 1e-12, atol: float = 1e-14) -> bool:
        return (
            allclose(self.R_EN, other.R_EN, rtol=rtol, atol=atol) and self.nvector == other.nvector
        )


class FrameL(FrameN):
    """
    Local level, Wander azimuth frame

    Attributes
    ----------
    nvector: Nvector
        Defines the origin of the local frame L. The origin is directly beneath or
        above the vehicle (B), at the surface of ellipsoid model.
    wander_azimuth: ndarray
        Angle(s) [rad] between the x-axis of L and the north direction.

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

    _NAMES = ("nvector", "wander_azimuth")
    wander_azimuth: NdArray

    def __init__(self, nvector: Nvector, wander_azimuth: ArrayLike = 0) -> None:
        """
        Initialize Local level, Wander azimuth frame from nvector

        Parameters
        ----------
        nvector: Nvector
            Position(s) of the vehicle (B) which also defines the origin of the local
            frame L. The origin is directly beneath or above the vehicle (B), at the
            surface of ellipsoid model.
        wander_azimuth: {array_like}
            Angle(s) [rad] between the x-axis of L and the north direction.
        """
        super().__init__(nvector)
        n = self.nvector.normal.shape[1]
        self.wander_azimuth = np.broadcast_to(np.asarray(wander_azimuth), n)

    @classmethod
    @format_docstring_types
    def from_point(
        cls, point: Union[ECEFvector, GeoPoint, Nvector], wander_azimuth: ArrayLike = 0
    ) -> "FrameL":
        """
        Returns FrameL with its origin projected from the point to the surface of ellipsoid model

        Parameters
        ----------
        point: ECEFvector, GeoPoint or Nvector
            Position of the vehicle (B) which also defines the origin of the local
            frame L. The origin is directly beneath or above the vehicle (B), at
            the surface of ellipsoid model.
        wander_azimuth: {array_like}
            Angle(s) [rad] between the x-axis of L and the north direction.
        """
        return cls(point.to_nvector(), wander_azimuth)

    @property
    def R_EN(self) -> NdArray:
        """Rotation matrix to go between E and L frame"""
        n_EA_E = self.nvector.normal
        R_Ee = self.nvector.frame.R_Ee
        return n_E_and_wa2R_EL(n_EA_E, self.wander_azimuth, R_Ee=R_Ee)


@use_docstring(_examples.get_examples_no_header([2]))
class FrameB(_LocalFrameBase):
    """
    Body frame

    Attributes
    ----------
    nvector: Nvector
        Position(s) of the vehicle's reference point which also coincides with
        the origin of the frame B.
    yaw, pitch, roll: ndarray
        Defining the orientation(s) of frame B in [rad].


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

    _NAMES = ("nvector", "yaw", "pitch", "roll")
    nvector: Nvector
    yaw: NdArray
    pitch: NdArray
    roll: NdArray

    def __init__(
        self,
        nvector: Nvector,
        yaw: ArrayLike = 0,
        pitch: ArrayLike = 0,
        roll: ArrayLike = 0,
        degrees: bool = False,
    ) -> None:
        """
        Initialize Body frame

        Parameters
        ----------
        nvector: Nvector
            Position of the vehicle's reference point which also coincides with
            the origin of the frame B.
        yaw, pitch, roll: {array_like}
            Defining the orientation of frame B in [deg] or [rad].
        degrees : bool
            if True yaw, pitch, roll are given in degrees otherwise in radians
        """
        self.nvector = nvector
        yaw_arr, pitch_arr, roll_arr = np.asarray(yaw), np.asarray(pitch), np.asarray(roll)
        if degrees:
            rad = np.deg2rad
            yaw_arr, pitch_arr, roll_arr = rad(yaw_arr), rad(pitch_arr), rad(roll_arr)
        n = self.nvector.normal.shape[1]
        self.yaw, self.pitch, self.roll = np.broadcast_arrays(
            yaw_arr, pitch_arr, roll_arr, np.ones(n)
        )[:3]

    @classmethod
    @format_docstring_types
    def from_point(
        cls,
        point: Union[ECEFvector, GeoPoint, Nvector],
        yaw: ArrayLike = 0,
        pitch: ArrayLike = 0,
        roll: ArrayLike = 0,
        degrees: bool = False,
    ) -> "FrameB":
        """
        Returns FrameB where its origin coincides with the vehicle's reference point.

        Parameters
        ----------
        point: ECEFvector, GeoPoint or Nvector
            Position of the vehicle's reference point which also coincides with
            the origin of the frame B.
        yaw, pitch, roll: {array_like}
            Defining the orientation(s) of frame B in [deg] or [rad].
        degrees : bool
            if True yaw, pitch, roll are given in degrees otherwise in radians
        """
        return cls(point.to_nvector(), yaw, pitch, roll, degrees)

    @property
    def R_EN(self) -> NdArray:
        """Rotation matrix to go between E and B frame"""
        R_NB = zyx2R(self.yaw, self.pitch, self.roll)
        n_EB_E = self.nvector.normal
        R_EN = n_E2R_EN(n_EB_E, self.nvector.frame.R_Ee)
        return mdot(R_EN, R_NB)  # rotation matrix

    def _is_equal_to(self, other: Any, rtol: float = 1e-12, atol: float = 1e-14) -> bool:
        return bool(
            allclose(self.yaw, other.yaw, rtol=rtol, atol=atol)
            and allclose(self.pitch, other.pitch, rtol=rtol, atol=atol)
            and allclose(self.roll, other.roll, rtol=rtol, atol=atol)
            and allclose(self.R_EN, other.R_EN, rtol=rtol, atol=atol)
            and self.nvector == other.nvector
        )


def _check_frames(
    obj1: Union[GeoPoint, Nvector, Pvector, ECEFvector],
    obj2: Union[GeoPoint, Nvector, Pvector, ECEFvector],
) -> None:
    if obj1.frame != obj2.frame:
        raise ValueError("Frames are unequal")


def _default_frame(
    frame: Optional[FrameE],
) -> FrameE:
    return frame if frame is not None else FrameE()


_ODICT = globals()
__doc__ = (
    __doc__  # @ReservedAssignment
    + _make_summary({n: _ODICT[n] for n in __all__})
    + ".. only:: draft\n\n"
    + "    License\n    -------\n    "
    + _license.__doc__.replace("\n", "\n    ")
)


if __name__ == "__main__":
    # print(__doc__)
    test_docstrings(__file__)
