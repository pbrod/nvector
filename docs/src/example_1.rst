Example 1: "A and B to delta"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex1img.png

Given two positions, A and B as latitudes, longitudes and depths relative to
Earth, E.

Find the exact vector between the two positions, given in meters north, east,
and down, and find the direction (azimuth) to B, relative to north.
Assume WGS-84 ellipsoid. The given depths are from the ellipsoid surface.
Use position A to define north, east, and down directions.
(Due to the curvature of Earth and different directions to the North Pole,
the north, east, and down directions will change (relative to Earth) for
different places.  A must be outside the poles for the north and east
directions to be defined.)

Solution:
    >>> import numpy as np
    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> pointA = wgs84.GeoPoint(latitude=1, longitude=2, z=3, degrees=True)
    >>> pointB = wgs84.GeoPoint(latitude=4, longitude=5, z=6, degrees=True)

Step 1: Find p_AB_E (delta decomposed in E).
    >>> p_AB_E = nv.diff_positions(pointA, pointB)

Step 2: Find p_AB_N (delta decomposed in N).
    >>> frame_N = nv.FrameN(pointA)
    >>> p_AB_N = p_AB_E.change_frame(frame_N)
    >>> p_AB_N = p_AB_N.pvector.ravel()
    >>> valtxt = '{0:8.2f}, {1:8.2f}, {2:8.2f}'.format(*p_AB_N)
    >>> 'Ex1: delta north, east, down = {}'.format(valtxt)
    'Ex1: delta north, east, down = 331730.23, 332997.87, 17404.27'

Step3: Also find the direction (azimuth) to B, relative to north:
    >>> azimuth = np.arctan2(p_AB_N[1], p_AB_N[0])
    >>> 'azimuth = {0:4.2f} deg'.format(np.rad2deg(azimuth))
    'azimuth = 45.11 deg'

Functional solution:
    >>> import numpy as np
    >>> import nvector as nv
    >>> from nvector import rad, deg

    >>> lat_EA, lon_EA, z_EA = rad(1), rad(2), 3
    >>> lat_EB, lon_EB, z_EB = rad(4), rad(5), 6

Step1: Convert to n-vectors:
    >>> n_EA_E = nv.lat_lon2n_E(lat_EA, lon_EA)
    >>> n_EB_E = nv.lat_lon2n_E(lat_EB, lon_EB)

Step2: Find p_AB_E (delta decomposed in E).WGS-84 ellipsoid is default:
    >>> p_AB_E = nv.n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA, z_EB)

Step3: Find R_EN for position A:
    >>> R_EN = nv.n_E2R_EN(n_EA_E)

Step4: Find p_AB_N (delta decomposed in N).
    >>> p_AB_N = np.dot(R_EN.T, p_AB_E).ravel()
    >>> valtxt = '{0:8.2f}, {1:8.2f}, {2:8.2f}'.format(*p_AB_N)
    >>> 'Ex1: delta north, east, down = {}'.format(valtxt)
    'Ex1: delta north, east, down = 331730.23, 332997.87, 17404.27'

Step5: Also find the direction (azimuth) to B, relative to north:
    >>> azimuth = np.arctan2(p_AB_N[1], p_AB_N[0])
    >>> 'azimuth = {0:4.2f} deg'.format(deg(azimuth))
    'azimuth = 45.11 deg'

See also `Example 1 at www.navlab.net <http://www.navlab.net/nvector/#example_1>`_ 

