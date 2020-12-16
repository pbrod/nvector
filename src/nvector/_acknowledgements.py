"""
The `nvector package <http://pypi.python.org/pypi/nvector/>`_ for
`Python <https://www.python.org/>`_ was written by Per A. Brodtkorb at
`FFI (The Norwegian Defence Research Establishment) <http://www.ffi.no/en>`_
based on the `nvector toolbox <http://www.navlab.net/nvector/#download>`_ for
`Matlab <http://www.mathworks.com>`_ written by the navigation group at
`FFI <http://www.ffi.no/en>`_. The nvector._core module is a vectorized reimplementation
of the matlab nvector toolbox while the nvector.objects module is a new easy to
use object oriented user interface to the nvector core functionality
documented in :cite:`GadeAndBrodtkorb2020Nvector`.

Most of the content is based on the article by K. Gade :cite:`Gade2010Nonsingular`.

Thus this article should be cited in publications using this page or
downloaded program code.

However, if you use any of the FrameE.direct, FrameE.inverse,
GeoPoint.distance_and_azimuth or GeoPoint.displace methods you should also cite the article by
Karney :cite:`Karney2013Algorithms` because these methods call Karney's
`geographiclib <https://pypi.python.org/pypi/geographiclib>`_ library to do the calculations.

"""