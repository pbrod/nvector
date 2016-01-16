
Example 2: "B and delta to C"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: http://www.navlab.net/images/ex2img.png

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
    >>> msg = 'Ex2: PosC: lat, lon = {:4.2f}, {:4.2f} deg,  height = {:4.2f} m'
    >>> msg.format(lat[0], lon[0], -z[0])
    'Ex2: PosC: lat, lon = 53.33, 63.47 deg,  height = 406.01 m'

See also `Example 2 at www.navlab.net <http://www.navlab.net/nvector/#example_2>`_ 
