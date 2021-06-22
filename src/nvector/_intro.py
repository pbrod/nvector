"""

.. only:: html

    |pkg_img| |tests_img| |docs_img| |health_img| |coverage_img| |versions_img| |downloads_img|


The nvector library is a suite of tools written in Python to solve geographical position
calculations. Currently the following operations are implemented:

* Calculate the surface distance between two geographical positions.

* Convert positions given in one reference frame into another reference frame.

* Find the destination point given start point, azimuth/bearing and distance.

* Find the mean position (center/midpoint) of several geographical positions.

* Find the intersection between two paths.

* Find the cross track distance between a path and a position.


Using n-vector, the calculations become simple and non-singular. 
Full accuracy is achieved for any global position (and for any distance).



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

n_E is n-vector in the program code, while in documents we use :math:`\mathbf{n}^{E}`. 
E denotes an Earth-fixed coordinate frame, and it indicates that the three components of
n-vector are along the three axes of E. More details about the notation and
reference frames can be found in the `documentation. 
<https://www.navlab.net/nvector/#vector_symbols>`_

"""