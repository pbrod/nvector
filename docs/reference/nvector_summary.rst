.. currentmodule:: nvector.objects


OO interface to Geodesic functions
----------------------------------
.. autosummary::
   :toctree: generated/

   delta_E
   delta_N
   delta_L
   diff_positions
   ECEFvector
   FrameB
   FrameE
   FrameN
   FrameL
   GeoPath
   GeoPoint
   Nvector
   Pvector

.. currentmodule:: nvector._core

Geodesic functions
------------------
.. autosummary::
   :toctree: generated/

    closest_point_on_great_circle
    cross_track_distance
    euclidean_distance
    great_circle_distance
    great_circle_normal
    interpolate
    intersect
    lat_lon2n_E
    mean_horizontal_position
    n_E2lat_lon
    n_EB_E2p_EB_E
    p_EB_E2n_EB_E
    n_EA_E_and_n_EB_E2p_AB_E
    n_EA_E_and_p_AB_E2n_EB_E
    n_EA_E_and_n_EB_E2azimuth
    n_EA_E_distance_and_azimuth2n_EB_E    
    on_great_circle
    on_great_circle_path

.. currentmodule:: nvector.rotation

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

.. currentmodule:: nvector.util

Utility functions
-----------------
.. autosummary::
   :toctree: generated/

   deg
   mdot
   nthroot
   rad
   get_ellipsoid
   select_ellipsoid
   unit

