Example 3: "ECEF-vector to geodetic latitude"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex3img.png

Position B is given as an "ECEF-vector" p_EB_E (i.e. a vector from E, the
center of the Earth, to B, decomposed in E).
Find the geodetic latitude, longitude and height (latEB, lonEB and hEB),
assuming WGS-84 ellipsoid.

Solution:
    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> position_B = 6371e3 * np.vstack((0.9, -1, 1.1))  # m
    >>> p_EB_E = wgs84.ECEFvector(position_B)
    >>> pointB = p_EB_E.to_geo_point()

    >>> lat, lon, h = pointB.latitude_deg, pointB.longitude_deg, -pointB.z
    >>> msg = 'Ex3: Pos B: lat, lon = {:4.2f}, {:4.2f} deg, height = {:9.2f} m'
    >>> msg.format(lat[0], lon[0], h[0])
    'Ex3: Pos B: lat, lon = 39.38, -48.01 deg, height = 4702059.83 m'

See also `Example 3 at www.navlab.net <http://www.navlab.net/nvector/#example_3>`_ 
