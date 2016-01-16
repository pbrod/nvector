Example 10: "Cross track distance"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex10img.png

Path A is given by the two positions A1 and A2 (similar to the previous
example).

Find the cross track distance sxt between the path A (i.e. the great circle
through A1 and A2) and the position B (i.e. the shortest distance at the
surface, between the great circle and B).

Also find the Euclidean distance dxt between B and the plane defined by the
great circle. Use Earth radius 6371e3.

Solution:
    >>> import nvector as nv
    >>> frame = nv.FrameE(a=6371e3, f=0)
    >>> pointA1 = frame.GeoPoint(0, 0, degrees=True)
    >>> pointA2 = frame.GeoPoint(10, 0, degrees=True)
    >>> pointB = frame.GeoPoint(1, 0.1, degrees=True)

    >>> pathA = nv.GeoPath(pointA1, pointA2)

    >>> s_xt = pathA.cross_track_distance(pointB, method='greatcircle').ravel()
    >>> d_xt = pathA.cross_track_distance(pointB, method='euclidean').ravel()
    >>> val_txt = '{:4.2f} km, {:4.2f} km'.format(s_xt[0]/1000, d_xt[0]/1000)
    >>> 'Ex10: Cross track distance: s_xt, d_xt = {}'.format(val_txt)
    'Ex10: Cross track distance: s_xt, d_xt = 11.12 km, 11.12 km'

See also `Example 10 at www.navlab.net <http://www.navlab.net/nvector/#example_10>`_ 
