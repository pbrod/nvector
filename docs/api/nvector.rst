nvector package
===============

.. currentmodule:: nvector._core

Geodesic functions
-------------------
.. autosummary::
   :toctree: generated/

    lat_lon2n_E
    n_E2lat_lon
    n_EB_E2p_EB_E
    p_EB_E2n_EB_E
    n_EA_E_and_n_EB_E2p_AB_E
    n_EA_E_and_p_AB_E2n_EB_E
    n_EA_E_and_n_EB_E2azimuth
    n_EA_E_distance_and_azimuth2n_EB_E
    great_circle_distance
    euclidean_distance
    mean_horizontal_position



Rotation matrices and angles
----------------------------
.. autosummary::
   :toctree: generated/

    n_E2R_EN
    n_E_and_wa2R_EL
    R_EL2n_E
    R_EN2n_E

    R2xyz
    R2zyx
    xyz2R
    zyx2R


Misc functions
---------------
.. autosummary::
   :toctree: generated/

   nthroot
   deg
   rad
   unit

.. currentmodule:: nvector.objects


OO interface to Geodesic functions
-----------------------------------
.. autosummary::
   :toctree: generated/

   FrameE
   FrameN
   FrameL
   FrameB
   ECEFvector
   GeoPoint
   Nvector
   GeoPath
   Pvector
	diff_positions

