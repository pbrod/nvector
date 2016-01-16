Example 7: "Mean position"
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex7img.png

Three positions A, B, and C are given as n-vectors n_EA_E, n_EB_E, and n_EC_E.
Find the mean position, M, given as n_EM_E.
Note that the calculation is independent of the depths of the positions.

Solution:
    >>> import nvector as nv
    >>> points = nv.GeoPoint(latitude=[90, 60, 50],
    ...                      longitude=[0, 10, -20], degrees=True)
    >>> nvectors = points.to_nvector()
    >>> n_EM_E = nvectors.mean_horizontal_position()
    >>> g_EM_E = n_EM_E.to_geo_point()
    >>> lat, lon = g_EM_E.latitude_deg, g_EM_E.longitude_deg
    >>> msg = 'Ex7: Pos M: lat, lon = {:4.2f}, {:4.2f} deg'
    >>> msg.format(lat[0], lon[0])
    'Ex7: Pos M: lat, lon = 67.24, -6.92 deg'

See also `Example 7 at www.navlab.net <http://www.navlab.net/nvector/#example_7>`_ 

