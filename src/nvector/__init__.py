from functools import wraps

from ._common import use_docstring
from ._info import __doc__ as __doc__  # @UnresolvedImport
from .core import (
    closest_point_on_great_circle,
    course_over_ground,
    cross_track_distance,
    euclidean_distance,
    geodesic_distance,
    geodesic_reckon,
    great_circle_distance,
    great_circle_distance_rad,
    great_circle_normal,
    interp_nvectors,
    interpolate,
    intersect,
    lat_lon2n_E,
    mean_horizontal_position,
    n_E2lat_lon,
    n_EA_E_and_n_EB_E2azimuth,
    n_EA_E_and_n_EB_E2p_AB_E,
    n_EA_E_and_n_EB_E2p_AB_N,
    n_EA_E_and_p_AB_E2n_EB_E,
    n_EA_E_and_p_AB_N2n_EB_E,
    n_EA_E_distance_and_azimuth2n_EB_E,
    n_EB_E2p_EB_E,
    on_great_circle,
    on_great_circle_path,
    p_EB_E2n_EB_E,
)
from .objects import (
    ECEFvector,
    FrameB,
    FrameE,
    FrameL,
    FrameN,
    GeoPath,
    GeoPoint,
    Nvector,
    Pvector,
    delta_E,
    delta_L,
    delta_N,
)
from .rotation import (
    E_rotation,
    R2xyz,
    R2zyx,
    R_EL2n_E,
    R_EN2n_E,
    change_axes_to_E,
    n_E2R_EN,
    n_E_and_wa2R_EL,
    xyz2R,
    zyx2R,
)
from .testing import test as _test  # noqa
from .util import (
    allclose,
    array_to_list_dict,
    deg,
    degrees2dm,
    dm2degrees,
    eccentricity2,
    get_ellipsoid,
    isclose,
    mdot,
    nthroot,
    polar_radius,
    rad,
    third_flattening,
    unit,
)

__version__ = "1.2.0"

_PACKAGE_NAME = __name__

__all__ = [
    # core
    "closest_point_on_great_circle",
    "course_over_ground",
    "cross_track_distance",
    "euclidean_distance",
    "geodesic_distance",
    "geodesic_reckon",
    "great_circle_distance",
    "great_circle_distance_rad",
    "great_circle_normal",
    "interp_nvectors",
    "interpolate",
    "intersect",
    "lat_lon2n_E",
    "mean_horizontal_position",
    "n_E2lat_lon",
    "n_EA_E_and_n_EB_E2azimuth",
    "n_EA_E_and_n_EB_E2p_AB_E",
    "n_EA_E_and_n_EB_E2p_AB_N",
    "n_EA_E_and_p_AB_E2n_EB_E",
    "n_EA_E_and_p_AB_N2n_EB_E",
    "n_EA_E_distance_and_azimuth2n_EB_E",
    "n_EB_E2p_EB_E",
    "on_great_circle",
    "on_great_circle_path",
    "p_EB_E2n_EB_E",
    # objects
    "ECEFvector",
    "FrameB",
    "FrameE",
    "FrameL",
    "FrameN",
    "GeoPath",
    "GeoPoint",
    "Nvector",
    "Pvector",
    "delta_E",
    "delta_L",
    "delta_N",
    # rotation
    "E_rotation",
    "R2xyz",
    "R2zyx",
    "R_EL2n_E",
    "R_EN2n_E",
    "change_axes_to_E",
    "n_E2R_EN",
    "n_E_and_wa2R_EL",
    "xyz2R",
    "zyx2R",
    # util
    "allclose",
    "array_to_list_dict",
    "deg",
    "degrees2dm",
    "dm2degrees",
    "eccentricity2",
    "get_ellipsoid",
    "isclose",
    "mdot",
    "nthroot",
    "polar_radius",
    "rad",
    "third_flattening",
    "unit",
]


@use_docstring(
    f"""
import {_PACKAGE_NAME} as {_PACKAGE_NAME[:2]}
{_PACKAGE_NAME[:2]}.test('-q', '--doctest-modules', '--cov={_PACKAGE_NAME}', '--disable-warnings')
"""
)
@wraps(_test)
def test(*options: str) -> int:
    return _test(__name__, *options)
