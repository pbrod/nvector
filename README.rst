=======
nvector
=======

.. image:: https://badge.fury.io/py/Nvector.png
    :target: https://pypi.python.org/pypi/Nvector/

.. image:: https://travis-ci.org/pbrod/Nvector.svg?branch=master
    :target: https://travis-ci.org/pbrod/Nvector

.. image:: https://readthedocs.org/projects/pip/badge/?version=latest
    :target: http://Nvector.readthedocs.org/en/latest/

.. image:: https://landscape.io/github/pbrod/Nvector/master/landscape.svg?style=flat
   :target: https://landscape.io/github/pbrod/Nvector/master
   :alt: Code Health

.. image:: https://coveralls.io/repos/pbrod/Nvector/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/pbrod/Nvector?branch=master

.. image:: https://img.shields.io/pypi/pyversions/Nvector.svg
   :target: https://github.com/pbrod/Nvector


Nvector is a suite of tools to solve geographical position calculations like:

* Calculate the surface distance between two geographical positions.

* Convert positions given in one reference frame into another reference frame.

* Find the destination point given start position, azimuth/bearing and distance.

* Find the mean position (center/midpoint) of several geographical positions.

* Find the intersection between two paths.

* Find the cross track distance between a path and a position.


Description
===========

In this library, we represent position with an "n-vector",  which
is the normal vector to the Earth model (the same reference ellipsoid that is
used for latitude and longitude). When using n-vector, all Earth-positions are
treated equally, and there is no need to worry about singularities or
discontinuities. An additional benefit with using n-vector is that many
position calculations can be solved with simple vector algebra
(e.g. dot product and cross product).

Converting between n-vector and latitude/longitude is unambiguous and easy
using the provided functions.


Methods
~~~~~~~

The core functions provided are:

- **lat_lon2n_E:** Converts latitude and longitude to n-vector.

- **n_E2lat_lon:** Converts n-vector to latitude and longitude.

- **n_EB_E2p_EB_E:** Converts n-vector to Cartesian position vector in meters.

- **p_EB_E2n_EB_E:** Converts Cartesian position vector in meters to n-vector.

- **n_EA_E_and_n_EB_E2p_AB_E:** From two positions A and B, finds the delta position.

- **n_EA_E_and_p_AB_E2n_EB_E:** From position A and delta, finds position B.

Nvector also provide an object oriented interface.

- **FrameE:**
    z-axis -> North, x-axis -> Latitude=Longitude=0
    Origo = Earth's centre.
    frame of reference rotates and moves with the Earth.
        
- **FrameN:**
    x-axis -> North, y-axis -> East, z-axis -> down
    Origo = Beneath/above Body at Earth's surface.

- **FrameL:**
    x-axis, y-axis -> wander azimuth, z-axis -> down
    Origo = Beneath/above Body at Earth's surface.

- **FrameB:**
    x-axis -> forward, y-axis -> starboard, z-axis -> body down    
    Origo = Body's centre.

- **ECEFvector:** Geographical position given as Cartesian position vector in frame E

- **GeoPoint:** Geographical position given as latitude, longitude, depth in frame E

- **Nvector:** Geographical position given as n-vector and depth in frame E

- **GeoPath:** Geodesic path between two points in Frame E


n_E is n-vector in the program code, while in documents we use nE. E denotes
an Earth-fixed coordinate frame, and it indicates that the three components of
n-vector are along the three axes of E. More details about the notation and reference frames can be found here:  

http://www.navlab.net/nvector/

www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf


Documentation and code
======================

Official documentation: http://www.navlab.net/nvector/

Bleeding edge: https://github.com/pbrod/nvector.

Official releases available at: http://pypi.python.org/pypi/nvector.


Installation and upgrade:
=========================

with pip

    $ pip install nvector


with easy_install

    $ easy_install nvector 

or

    $ easy_install upgrade nvector

to upgrade to the newest version


Unit tests
===========
To test if the toolbox is working paste the following in an interactive
python session::

   import nvector as nv
   nv.test(coverage=True, doctests=True)


Acknowledgement
===============

Written by the navigation group at FFI (The Norwegian Defence Research Establishment). 

Most of the content is based on the following article:

Kenneth Gade (2010): A Nonsingular Horizontal Position Representation, The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010. www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf

Thus this article should be cited in publications using this page or the downloaded program code.


Getting Started
===============

Below the object-oriented solution to some common geodesic problems are given.
In the first example the functional solution is also given.
The functional solutions to the remaining problems can be found 
`here <https://github.com/pbrod/nvector/blob/master/nvector/tests/test_nvector.py>`_.


Example 1: "A and B to delta"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    >>> p_AB_E = diff_positions(pointA, pointB)

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
    >>> 'delta north, east, down = {0:8.2f}, {1:8.2f}, {2:8.2f}'.format(*p_AB_N)
    'delta north, east, down = 331730.23, 332997.87, 17404.27'

Step5: Also find the direction (azimuth) to B, relative to north:
    >>> azimuth = np.arctan2(p_AB_N[1], p_AB_N[0]) # positive angle about down-axis
    >>> 'azimuth = {0:4.2f} deg'.format(deg(azimuth))
    'azimuth = 45.11 deg'



Example 2: "B and delta to C"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A radar or sonar attached to a vehicle B (Body coordinate frame) measures the
distance and direction to an object C. We assume that the distance and two
angles (typically bearing and elevation relative to B) are already combined to
the vector p_BC_B (i.e. the vector from B to C, decomposed in B). The position
of B is given as n_EB_E and z_EB, and the orientation (attitude) of B is given
as R_NB (this rotation matrix can be found from roll/pitch/yaw by using zyx2R).

Find the exact position of object C as n-vector and depth ( n_EC_E and z_EC ),
assuming Earth ellipsoid with semi-major axis a and flattening f. For WGS-72,
use a = 6 378 135 m and f = 1/298.26.

Solution:
    >>> import nvector as nv
    >>> wgs72 = nv.FrameE(name='WGS72')
    >>> wgs72 = nv.FrameE(a=6378135, f=1.0/298.26)

Step 1: Position and orientation of B is given 400m above E:
    >>> n_EB_E = wgs72.Nvector(nv.unit([[1], [2], [3]]), z=-400)

Step 2: Delta BC decomposed in B
    >>> frame_B = nv.FrameB(n_EB_E, yaw=10, pitch=20, roll=30, degrees=True)
    >>> p_BC_B = frame_B.Pvector(np.r_[3000, 2000, 100].reshape((-1, 1)))

Step 3: Decompose delta BC in E
    >>> p_BC_E = p_BC_B.to_ecef_vector()

Step 4: Find point C by adding delta BC to EB
    >>> p_EB_E = n_EB_E.to_ecef_vector()
    >>> p_EC_E = p_EB_E + p_BC_E
    >>> pointC = p_EC_E.to_geo_point()

    >>> lat, lon, z = pointC.latitude_deg, pointC.longitude_deg, pointC.z
    >>> msg = 'Ex2: Pos C: lat, lon = {:4.2f}, {:4.2f} deg,  height = {:4.2f} m'
    >>> msg.format(lat[0], lon[0], -z[0])
    'Ex2: Pos C: lat, lon = 53.33, 63.47 deg,  height = 406.01 m'


Example 3: "ECEF-vector to geodetic latitude"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Example 4: "Geodetic latitude to ECEF-vector"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Solution:
    >>> import nvector as nv
    >>> wgs84 = nv.FrameE(name='WGS84')
    >>> pointB = wgs84.GeoPoint(latitude=1, longitude=2, z=-3, degrees=True)
    >>> p_EB_E = pointB.to_ecef_vector()

    >>> 'Ex4: p_EB_E = {} m'.format(p_EB_E.pvector.ravel())
    'Ex4: p_EB_E = [ 6373290.27721828   222560.20067474   110568.82718179] m'


Example 5: "Surface distance"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Find the surface distance sAB (i.e. great circle distance) between two positions A and B. 
The heights of A and B are ignored, i.e. if 
they don't have zero height, we seek the distance between the points that are at the 
surface of the Earth, directly above/below A and B. The Euclidean distance (chord length)
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

    >>> msg = 'Ex5: Great circle and Euclidean distance = {:5.2f} km, {:5.2f} km'
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


Example 7: "Mean position"
~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Example 8: "A and azimuth/distance to B"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    >>> latB, lonB = pointB.latitude_deg, pointB.longitude_deg

    >>> 'Ex8, Destination: lat, lon = {:4.2f} deg, {:4.2f} deg'.format(latB, lonB)
    'Ex8, Destination: lat, lon = 79.99 deg, -90.02 deg'


Example 9: "Intersection of two paths"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


Example 10: "Cross track distance"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    >>> msg = 'cross track distance from path A to position B'
    >>> '{}, s_xt, d_xt = {}'.format(msg, val_txt)
    'cross track distance from path A to position B, s_xt, d_xt = 11.12 km, 11.12 km'


See also
--------
`geographiclib <https://pypi.python.org/pypi/geographiclib>`_


Note
====

This project has been set up using PyScaffold 2.4.4. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
