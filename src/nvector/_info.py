from nvector._examples import getting_started
__doc__ = """
Introduction to Nvector
=======================

|nvector_img| |tests_img| |docs_img| |health_img| |coverage_img| |versions_img|

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

Documentation and code
======================

Official documentation:

http://www.navlab.net/nvector/

http://nvector.readthedocs.io/en/latest/

*Kenneth Gade (2010):*
    `A Nonsingular Horizontal Position Representation,
    The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
    <http://www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>`_


Bleeding edge: https://github.com/pbrod/nvector.

Official releases available at: http://pypi.python.org/pypi/nvector.


Installation
============

If you have pip installed and are online, then simply type:

    $ pip install nvector

to get the lastest stable version. Using pip also has the advantage that all
requirements are automatically installed.

You can download nvector and all dependencies to a folder "pkg", by the following:

   $ pip install --download=pkg nvector

To install the downloaded nvector, just type:

   $ pip install --no-index --find-links=pkg nvector


Unit tests
===========
To test if the toolbox is working paste the following in an interactive
python session::

   import nvector as nv
   nv.test()


Acknowledgement
===============
The `nvector package <http://pypi.python.org/pypi/nvector/>`_ for
`Python <https://www.python.org/>`_ was written by Per A. Brodtkorb at
`FFI (The Norwegian Defence Research Establishment) <http://www.ffi.no/en>`_
based on the `nvector toolbox <http://www.navlab.net/nvector/#download>`_ for
`Matlab <http://www.mathworks.com>`_ written by the navigation group at
`FFI <http://www.ffi.no/en>`_.

Most of the content is based on the following article:

*Kenneth Gade (2010):*
    `A Nonsingular Horizontal Position Representation,
    The Journal of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
    <http://www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf>`_

Thus this article should be cited in publications using this page or the
downloaded program code.

""" + getting_started + """
See also
========
`geographiclib <https://pypi.python.org/pypi/geographiclib>`_

.. |nvector_img| image:: https://badge.fury.io/py/Nvector.png
   :target: https://pypi.python.org/pypi/Nvector/
.. |tests_img| image:: https://travis-ci.org/pbrod/Nvector.svg?branch=master
   :target: https://travis-ci.org/pbrod/Nvector
.. |docs_img| image:: https://readthedocs.org/projects/pip/badge/?version=stable
   :target: http://Nvector.readthedocs.org/en/stable/
.. |health_img| image:: https://landscape.io/github/pbrod/Nvector/master/landscape.svg?style=flat
   :target: https://landscape.io/github/pbrod/Nvector/master
.. |coverage_img| image:: https://coveralls.io/repos/pbrod/Nvector/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/pbrod/Nvector?branch=master
.. |versions_img| image:: https://img.shields.io/pypi/pyversions/Nvector.svg
   :target: https://github.com/pbrod/Nvector

"""


if __name__ == '__main__':
    from nvector._common import write_readme, test_docstrings
    test_docstrings(__file__)
    write_readme(__doc__)
