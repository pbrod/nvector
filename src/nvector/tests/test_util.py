"""
Created on 23. jun. 2021

@author: pab
"""
from functools import partial
import pytest
import numpy as np
from hypothesis import given, strategies as st
from numpy.testing import assert_allclose as _assert_allclose  # @UnresolvedImport
from nvector.util import deg, rad, mdot, nthroot, get_ellipsoid, unit, Ellipsoid

assert_allclose = partial(_assert_allclose, atol=1e-15)


@pytest.mark.parametrize(
    "names, datum",
    [(("airy1858", 1),
      Ellipsoid(a=6377563.3960, f=1.0 / 299.3249646, name="Airy 1858")),
     (("airymodified", 2),
      Ellipsoid(a=6377340.189, f=1.0 / 299.3249646, name="Airy Modified")),
     (("australiannational", 3),
      Ellipsoid(a=6378160.0, f=1.0 / 298.25, name="Australian National")),
     (("bessel", "bessel1841", 4),
      Ellipsoid(a=6377397.155, f=1.0 / 299.1528128, name="Bessel 1841")),
     (("clarke1880", 5),
      Ellipsoid(a=6378249.145, f=1.0 / 293.465, name="Clarke 1880")),
     (("everest1830", 6),
      Ellipsoid(a=6377276.345, f=1.0 / 300.8017, name="Everest 1830")),
     (("everestmodified", 7),
      Ellipsoid(a=6377304.063, f=1.0 / 300.8017, name="Everest Modified")),
     (("fisher1960", 8),
      Ellipsoid(a=6378166.0, f=1.0 / 298.3, name="Fisher 1960")),
     (("fisher1968", 9),
      Ellipsoid(a=6378150.0, f=1.0 / 298.3, name="Fisher 1968")),
     (("hough1956", "hough", 10),
      Ellipsoid(a=6378270.0, f=1.0 / 297, name="Hough 1956")),
     (("international", "hayford", "ed50", 11),
      Ellipsoid(a=6378388.0, f=1.0 / 297,
                name="Hayford/International ellipsoid 1924/European Datum 1950/ED50")),
     (("krassovsky", "krassovsky1938", 12),
      Ellipsoid(a=6378245.0, f=1.0 / 298.3, name="Krassovsky 1938")),
     (("nwl-9d", "wgs66", 13),
      Ellipsoid(a=6378145.0, f=1.0 / 298.25, name="NWL-9D / WGS 66")),
     (("southamerican1969", "sad69", 14),
      Ellipsoid(a=6378160.0, f=1.0 / 298.25, name="South American 1969 / SAD69")),
     (("sovietgeod.system1985", 15),
      Ellipsoid(a=6378136.0, f=1.0 / 298.257, name="Soviet Geod. System 1985")),
     (("wgs72", 16),
      Ellipsoid(a=6378135.0, f=1.0 / 298.26, name="WGS 72")),
     (("clarke1866", "nad27", 17),
      Ellipsoid(a=6378206.4, f=1.0 / 294.9786982138, name="Clarke 1866 / NAD27")),
     (("grs80", "wgs84", "nad83", 18),
      Ellipsoid(a=6378137.0, f=1.0 / 298.257223563, name="GRS80 / WGS84 / NAD83")),
     (("euref89", "etrs89", 19),
      Ellipsoid(a=6378137.0, f=298.257222101, name="ETRS89 / EUREF89")),
     (("ngo1948", 20),
      Ellipsoid(a=6377492.0176, f=1 / 299.15281285, name="NGO1948"))
     ])
def test_get_ellipsoid(names, datum):
    for name in names:
        ellipsoid = get_ellipsoid(name)
        assert ellipsoid == datum
    for nick_name in datum.name.split("/"):
        ellipsoid = get_ellipsoid(nick_name)
        assert ellipsoid == datum

    ellipsoid = get_ellipsoid(datum.name)
    assert ellipsoid == datum


@given(st.floats())
def test_rad_and_deg(values):
    radians = rad(values)
    assert_allclose(deg(radians), values)


@given(st.floats(min_value=0.0, max_value=1e15), st.integers(min_value=1, max_value=10))
def test_nthroot(value, n):
    nvalue = value**n
    assert_allclose(nthroot(nvalue, n), value)


@given(st.floats(), st.floats(), st.floats())
def test_unit(x, y, z):
    vec = unit([[x], [y], [z]])
    if np.all(np.abs([x, y, z]) < np.inf):
        assert_allclose(np.sqrt(np.sum(vec**2)), 1.0)
    else:
        assert np.all(np.isnan(vec))


def test_mdot():
    a = 1.0 * np.arange(18).reshape(3, 3, 2)
    b = - a
    t = np.concatenate([np.dot(a[..., i], b[..., i])[:, :, None]
                        for i in range(2)], axis=2)
    tm = mdot(a, b)

    assert_allclose(t, tm)

    t1 = np.concatenate([np.dot(a[..., i], b[:, 0, 0][:, None])[:, :, None]
                         for i in range(2)], axis=2)

    tm1 = mdot(a, b[:, 0, 0].reshape(-1, 1))

    assert_allclose(t1, tm1)

    tt0 = mdot(a[..., 0], b[..., 0])

    assert_allclose(t[..., 0], tt0)

    tt0 = mdot(a[..., 0], b[:, :1, 0])

    assert_allclose(t[:, :1, 0], tt0)

    tt0 = mdot(a[..., 0], b[:, :2, 0][:, None])

    assert_allclose(t[:, :2, 0][:, None], tt0)
