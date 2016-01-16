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


Nvector is a suite of tools written in Python to solve geographical position
calculations like:

* Calculate the surface distance between two geographical positions.

* Convert positions given in one reference frame into another reference frame.

* Find the destination point given start point, azimuth/bearing and distance.

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

n_E is n-vector in the program code, while in documents we use nE. E denotes
an Earth-fixed coordinate frame, and it indicates that the three components of
n-vector are along the three axes of E. More details about the notation and
reference frames can be found here:

http://www.navlab.net/nvector/

http://www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf


Methods
~~~~~~~

The core functions provided are:

- **lat_lon2n_E:**
    Converts latitude and longitude to n-vector.
- **n_E2lat_lon:**
    Converts n-vector to latitude and longitude.
- **n_EB_E2p_EB_E:**
    Converts n-vector to Cartesian position vector in meters.
- **p_EB_E2n_EB_E:**
    Converts Cartesian position vector in meters to n-vector.
- **n_EA_E_and_n_EB_E2p_AB_E:**
    From two positions A and B, finds the delta position.
- **n_EA_E_and_p_AB_E2n_EB_E:** 
    From position A and delta, finds position B.


Nvector also provide an object oriented interface.

- **FrameE:**
    z-axis -> North Pole, x-axis -> Latitude=Longitude=0.
    Origo = Earth's centre.
- **FrameN:**
    x-axis -> North, y-axis -> East, z-axis -> down.
    Origo = Beneath/above Body at Earth's surface.
- **FrameL:**
    x-axis, y-axis -> wander azimuth, z-axis -> down.
    Origo = Beneath/above Body at Earth's surface.
- **FrameB:**
    x-axis -> forward, y-axis -> starboard, z-axis -> body down.
    Origo = Body's centre.
- **ECEFvector:**
    Geographical position given as Cartesian position vector in frame E
- **GeoPoint:**
    Geographical position given as latitude, longitude, depth in frame E
- **Nvector:**
    Geographical position given as n-vector and depth in frame E
- **GeoPath:**
    Geodesic path between two points in frame E



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
The `nvector package <http://pypi.python.org/pypi/nvector/>`_ for `Python <https://www.python.org/>`_ 
was written by Per A. Brodtkorb at `FFI (The Norwegian Defence Research Establishment) <http://www.ffi.no/en>`_ 
based on the `nvector toolbox <http://www.navlab.net/nvector/#download>`_ for 
`Matlab <http://www.mathworks.com>`_ written by the navigation group at `FFI <http://www.ffi.no/en>`_.

Most of the content is based on the following article:

*Kenneth Gade (2010):*
    `A Nonsingular Horizontal Position Representation,
    The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
    <http://www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>`_

Thus this article should be cited in publications using this page or the
downloaded program code.


.. include:: ./docs/getting_started.rst

Note
====

This project has been set up using PyScaffold 2.4.4. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.