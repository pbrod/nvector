Example 9: "Intersection of two paths"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex9img.png

Define a path from two given positions (at the surface of a spherical Earth),
as the great circle that goes through the two points.

Path A is given by A1 and A2, while path B is given by B1 and B2.

Find the position C where the two paths intersect.

Solution:
    >>> import nvector as nv
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

See also `Example 9 at www.navlab.net <http://www.navlab.net/nvector/#example_9>`_ 
