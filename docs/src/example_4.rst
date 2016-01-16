Example 4: "Geodetic latitude to ECEF-vector"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex4img.png

Geodetic latitude, longitude and height are given for position B as latEB, longEB and hEB, find the ECEF-vector for this position, p_EB_E.

Solution:
    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> pointB = wgs84.GeoPoint(latitude=1, longitude=2, z=-3, degrees=True)
    >>> p_EB_E = pointB.to_ecef_vector()

    >>> 'Ex4: p_EB_E = {} m'.format(p_EB_E.pvector.ravel())
    'Ex4: p_EB_E = [ 6373290.27721828   222560.20067474   110568.82718179] m'

See also `Example 4 at www.navlab.net <http://www.navlab.net/nvector/#example_4>`_ 
