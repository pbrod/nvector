Example 8: "A and azimuth/distance to B"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex8img.png

We have an initial position A, direction of travel given as an azimuth
(bearing) relative to north (clockwise), and finally the
distance to travel along a great circle given as sAB.
Use Earth radius 6371e3 m to find the destination point B.

In geodesy this is known as "The first geodetic problem" or
"The direct geodetic problem" for a sphere, and we see that this is similar to
Example 2, but now the delta is given as an azimuth and a great circle
distance. ("The second/inverse geodetic problem" for a sphere is already
solved in Examples 1 and 5.)

Solution:
    >>> import nvector as nv
    >>> frame = nv.FrameE(a=6371e3, f=0)
    >>> pointA = frame.GeoPoint(latitude=80, longitude=-90, degrees=True)
    >>> pointB, _azimuthb = pointA.geo_point(distance=1000, azimuth=200,
    ...                                      degrees=True)
    >>> lat, lon = pointB.latitude_deg, pointB.longitude_deg

    >>> msg = 'Ex8, Destination: lat, lon = {:4.2f} deg, {:4.2f} deg'
    >>> msg.format(lat, lon)
    'Ex8, Destination: lat, lon = 79.99 deg, -90.02 deg'

See also `Example 8 at www.navlab.net <http://www.navlab.net/nvector/#example_8>`_ 
