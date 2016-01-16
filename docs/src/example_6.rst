Example 6 "Interpolated position"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex6img.png

Given the position of B at time t0 and t1, n_EB_E(t0) and n_EB_E(t1).

Find an interpolated position at time ti, n_EB_E(ti). All positions are given
as n-vectors.

Solution:
    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> path = nv.GeoPath(wgs84.GeoPoint(89, 0, degrees=True),
    ...                   wgs84.GeoPoint(89, 180, degrees=True))

    >>> t0 = 10.
    >>> t1 = 20.
    >>> ti = 16.  # time of interpolation
    >>> ti_n = (ti - t0) / (t1 - t0) # normalized time of interpolation

    >>> g_EB_E_ti = path.interpolate(ti_n).to_geo_point()

    >>> lat_ti, lon_ti = g_EB_E_ti.latitude_deg, g_EB_E_ti.longitude_deg
    >>> msg = 'Ex6, Interpolated position: lat, long = {} deg, {} deg'
    >>> msg.format(lat_ti, lon_ti)
    'Ex6, Interpolated position: lat, long = [ 89.7999805] deg, [ 180.] deg'

See also `Example 6 at www.navlab.net <http://www.navlab.net/nvector/#example_6>`_ 
