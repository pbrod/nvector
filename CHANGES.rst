=========
Changelog
=========

Created with gitcommand: git shortlog v0.0.0..v0.1.1

Version 0.4.1, Januar 14, 2016
==============================

pbrod (63):

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
      *Added links to FFI homepage.
      * Updated documentation:     Added link to nvector toolbox for matlab     For each example added links to the more detailed explanation on the homepage
      * Updated link to nvector toolbox for matlab
      * Added link to nvector on  pypi
      * Updated documentation fro FrameB, FrameE, FrameL and FrameN.
      * updated __all__ variable


Version 0.3.1, Januar 1, 2016
=============================

pbrod (31):

 		* Refactored
      * Updated tests
      * Updated docs
      * Moved tests to nvector/tests
      * Merge https://github.com/pbrod/Nvector
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


Version 0.1.1, Januar 1, 2016
=============================

pbrod (31):
      * Initial commit: Translated code from Matlab to Python.
      * Added object oriented interface to nvector library
      * Added tests for object oriented interface
      * Added geodesic tests.