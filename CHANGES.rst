=========
Changelog
=========

Version 0.6.0, December 09, 2018
================================
Per A Brodtkorb (79):
      * Updated requirements in setup.py
      * Removed tox.ini
      * Updated documentation on how to set package version
      * Made a separate script to set package version in nvector/__init__.py
      * Updated docstring for select_ellipsoid
      * Replace GeoPoint.geo_point with GeoPoint.displace and removed deprecated GeoPoint.geo_point
      * Update .travis.yml
      * Fix so that codeclimate is able to parse .travis.yml
      * Only run sonar and codeclimate reporter for python v3.6
      * Added sonar-project.properties
      * Pinned coverage to v4.3.4 due to fact that codeclimate reporter is only
         compatible with Coverage.py versions >=4.0,<4.4.
      * Updated with sonar scanner.
      * Added .pylintrc
      * Set up codeclimate reporter
      * Updated docstring for unit function.
      * Avoid division by zero in unit function.
      * Reenabled the doctest of plot_mean_position
      * Reset "pyscaffold==2.5.11"
      * Replaced deprecated basemap with cartopy.
      * Replaced doctest of plot_mean_position with test_plot_mean_position in
         test_plot.py
      * Fixed failing doctests for python v3.4 and v3.5 and made them more
         robust.
      * Fixed failing doctests and made them more robust.
      * Increased pycoverage version to use.
      * moved nvector to src/nvector/
      * Reset the setup.py to require 'pyscaffold==2.5.11' which works on
         python version 3.4, 3.5 and 3.6. as well as 2.7
      * Updated unittests.
      * Updated tests.
      * Removed obsolete code
      * Added test for delta_L
      * Added corner testcase for
         pointA.displace(distance=1000,azimuth=np.deg2rad(200))
      * Added test for path.track_distance(method='exact')
      * Added delta_L a function thet teturn cartesian delta vector from
         positions A to B decomposed in L.
      * Simplified OO-solution in example 1 by using delta_N function
      * Refactored duplicated code
      * Vectorized code so that the frames can take more than one position at
         the time.
      * Keeping only the html docs in the distribution.
      * replaced link from latest to stable docs on readthedocs and updated
         crosstrack distance test.
      * updated documentation in setup.py

Version 0.5.2, March 7, 2017
============================


Per A Brodtkorb (10):
      * Fixed tests in tests/test_frames.py
      * Updated to setup.cfg and tox.ini + pep8
      * updated .travis.yml
      * Updated Readme.rst with new example 10 picture and link to nvector docs at readthedocs.
      * updated official documentation links
      * Updated crosstrack distance tests.


Version 0.5.1, March 5, 2017
============================


Cody (4):
     * Explicitely numbered replacement fields
     * Migrated `%` string formating

Per A Brodtkorb (29):
     * pep8
     * Updated failing examples
     * Updated README.rst
     * Removed obsolete pass statement
     * Documented functions
     * added .checkignore for quantifycode
     * moved test_docstrings and use_docstring_from into _common.py
     * Added .codeclimate.yml
     * Updated installation information in _info.py
     * Added GeoPath.on_path method. Clearified intersection example
     * Added great_circle_normal, cross_track_distance Renamed intersection to intersect (Intersection is deprecated.)
     * Simplified R2zyx with a call to R2xyz Improved accuracy for great circle cross track distance for small distances.
     * Added on_great_circle, _on_great_circle_path, _on_ellipsoid_path, closest_point_on_great_circle and closest_point_on_path to GeoPath
     * made __eq__ more robust for frames
     * Removed duplicated code
     * Updated tests
     * Removed fishy test
     * replaced zero n-vector with nan
     * Commented out failing test.
     * Added example 10 image
     * Added 'closest_point_on_great_circle', 'on_great_circle','on_great_circle_path'.
     * Updated examples + documentation
     * Updated index depth
     * Updated README.rst and classifier in setup.cfg



Version 0.4.1, January 19, 2016
===============================

pbrod (46):

      * Cosmetic updates
      * Updated README.rst
      * updated docs and removed unused code
      * updated README.rst and .coveragerc
      * Refactored out _check_frames
      * Refactored out _default_frame
      * Updated .coveragerc
      * Added link to geographiclib
      * Updated external link
      * Updated documentation
      * Added figures to examples
      * Added GeoPath.interpolate + interpolation example 6
      * Added links to FFI homepage.
      * Updated documentation:
          - Added link to nvector toolbox for matlab
          - For each example added links to the more detailed explanation on the homepage
      * Updated link to nvector toolbox for matlab
      * Added link to nvector on  pypi
      * Updated documentation fro FrameB, FrameE, FrameL and FrameN.
      * updated __all__ variable
      * Added missing R_Ee to function n_EA_E_and_n_EB_E2azimuth + updated documentation
      * Updated CHANGES.rst
      * Updated conf.py
      * Renamed info.py to _info.py
      * All examples are now generated from _examples.py.


Version 0.1.3, January 1, 2016
==============================

pbrod (31):

      * Refactored
      * Updated tests
      * Updated docs
      * Moved tests to nvector/tests
      * Updated .coverage     Added travis.yml, .landscape.yml
      * Deleted obsolete LICENSE
      * Updated README.rst
      * Removed ngs version
      * Fixed bug in .travis.yml
      * Updated .travis.yml
      * Removed dependence on navigator.py
      * Updated README.rst
      * Updated examples
      * Deleted skeleton.py and added tox.ini
      * Small refactoring     Renamed distance_rad_bearing_rad2point to n_EA_E_distance_and_azimuth2n_EB_E     updated tests
      * Renamed azimuth to n_EA_E_and_n_EB_E2azimuth     Added tests for R2xyz as well as R2zyx
      * Removed backward compatibility     Added test_n_E_and_wa2R_EL
      * Refactored tests
      * Commented out failing tests on python 3+
      * updated CHANGES.rst
      * Removed bug in setup.py


Version 0.1.1, January 1, 2016
==============================

pbrod (31):
      * Initial commit: Translated code from Matlab to Python.
      * Added object oriented interface to nvector library
      * Added tests for object oriented interface
      * Added geodesic tests.
