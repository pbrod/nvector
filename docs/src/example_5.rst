Example 5: "Surface distance"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex5img.png

Find the surface distance sAB (i.e. great circle distance) between two
positions A and B. The heights of A and B are ignored, i.e. if they don't have
zero height, we seek the distance between the points that are at the surface of
the Earth, directly above/below A and B. The Euclidean distance (chord length)
dAB should also be found. Use Earth radius 6371e3 m.
Compare the results with exact calculations for the WGS-84 ellipsoid.

Great circle solution:
    >>> import nvector as nv
    >>> frame_E = nv.FrameE(a=6371e3, f=0)
    >>> positionA = frame_E.GeoPoint(latitude=88, longitude=0, degrees=True)
    >>> positionB = frame_E.GeoPoint(latitude=89, longitude=-170, degrees=True)

    >>> s_AB, _azia, _azib = positionA.distance_and_azimuth(positionB)
    >>> p_AB_E = positionB.to_ecef_vector() - positionA.to_ecef_vector()
    >>> d_AB = np.linalg.norm(p_AB_E.pvector, axis=0)[0]

    >>> msg = 'Ex5: Great circle and Euclidean distance = {}'
    >>> msg = msg.format('{:5.2f} km, {:5.2f} km')
    >>> msg.format(s_AB / 1000, d_AB / 1000)
    'Ex5: Great circle and Euclidean distance = 332.46 km, 332.42 km'

Alternative great circle solution:
    >>> path = nv.GeoPath(positionA, positionB)
    >>> s_AB2 = path.track_distance(method='greatcircle').ravel()
    >>> d_AB2 = path.track_distance(method='euclidean').ravel()
    >>> msg.format(s_AB2[0] / 1000, d_AB2[0] / 1000)
    'Ex5: Great circle and Euclidean distance = 332.46 km, 332.42 km'

Exact solution for the WGS84 ellipsoid:
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> point1 = wgs84.GeoPoint(latitude=88, longitude=0, degrees=True)
    >>> point2 = wgs84.GeoPoint(latitude=89, longitude=-170, degrees=True)
    >>> s_12, _azi1, _azi2 = point1.distance_and_azimuth(point2)

    >>> p_12_E = point2.to_ecef_vector() - point1.to_ecef_vector()
    >>> d_12 = np.linalg.norm(p_12_E.pvector, axis=0)[0]
    >>> msg = 'Ellipsoidal and Euclidean distance = {:5.2f} km, {:5.2f} km'
    >>> msg.format(s_12 / 1000, d_12 / 1000)
    'Ellipsoidal and Euclidean distance = 333.95 km, 333.91 km'

See also `Example 5 at www.navlab.net <http://www.navlab.net/nvector/#example_5>`_ 
