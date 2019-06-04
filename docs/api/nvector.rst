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
    cross_track_distance
    closest_point_on_great_circle
    intersect
    mean_horizontal_position
    on_great_circle
    on_great_circle_path


Rotation matrices and angles
----------------------------
.. autosummary::
   :toctree: generated/
	
    E_rotation
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
   select_ellipsoid
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
   delta_E
   delta_N
   delta_L

