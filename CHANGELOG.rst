=========
Changelog
=========

Version 0.7.6, December 17, 2020
================================

Per A Brodtkorb (26):
      * Added missing functions great_circle_normal and interpolate to the nvector_summary.rst
      * Moved the following functions related to rotation matrices from _core to rotation module:   
         - E_rotation, n_E_and_wa2R_EL, n_E2R_EN, R_EL2n_E, R_EN2n_E, R2xyz, R2zyx, xyz2R, zyx2R
      * Renamed select_ellipsoid to get_ellipsoid 
      * Moved the following utility functions from _core to util module:   
         - deg, rad, mdot, nthroot, get_ellipsoid, unit, _check_length_deviation 
      * Added _get_h1line and _make_summary to _common.py 
      * Replaced numpy.rollaxis with numpy.swapaxes to make the code clearer.
      * _atleast_3d now broadcast the input against each other.
      * Added examples to zyx2R 
      * Added the following references to zyx2R, xyz2R, R2xyz, R2zyx: 
         - https://en.wikipedia.org/wiki/Aircraft_principal_axes
         - https://en.wikipedia.org/wiki/Euler_angles
         - https://en.wikipedia.org/wiki/Axes_conventions
      * Removed tabs from CHANGELOG.rst
      * Updated CHANGELOG.rst and prepared for release v0.7.6
      * Fixed the documentation so that it shows correctly in the reference manual. 
      * Added logo.png and docs/reference/nvector.rst
      * Updated build_package.py so it generates a valid README.rst file.
      * Updated THANKS.rst
      * Updated CHANGELOG.rst and prepare for release 0.7.6
      * Added Nvector documentation ref https://nvector.readthedocs.io/en/v0.7.5 to refs1.bib and _acknowledgements.py
      * Updated README.rst
      * Renamed requirements.readthedocs.txt to docs/requirements.txt 
      * Added .readthedocs.yml
      * Added sphinxcontrib-bibtex to requirements.readthedocs.txt
      * Added missing docs/tutorials/images/ex3img.png 
      * Deleted obsolete ex10img.png 
      * Updated acknowledgement with reference to Karney's article.
      * Updated README.rst by moving acknowledgement to the end with references.
      * Renamed position input argument to point in the FrameN, FrameB and FrameL classes. 
      * Deleted _example_images.py
      * Renamed nvector.rst to nvector_summary.rst in docs/reference
      * Added example images to tutorials/images/ folder 
      * Added Nvector logo, install.rst to docs 
      * Added src/nvector/_example_images.py
      * Added docs/tutorials/whatsnext.rst
      * Reorganized the documentation in docs by splitting _info.py into: 
          - _intro.py, 
          - _documentation.py
          - _examples_object_oriented.py
          - _images.py
          - _installation.py and _acknowledgements.py   
      * Added docs/tutorials/index.rst, docs/intro/index.rst, docs/how-to/index.rst docs/appendix/index.rst and docs/make.bat
      * updated references.


Version 0.7.5, December 12, 2020
================================

Per A Brodtkorb (32):
      * Updated CHANGELOG.rst and prepare for release 0.7.5
      * Changed so that GeoPath.on_great_circle and GeoPath.on_great_circle
         returns scalar result if the two points defining the path are scalars. See issue #10.
      * Fixed failing doctests.
      * Added doctest configuration to docs/conf.py
      * Added allclose to nvector/objects.py
      * Added array_to_list_dict and isclose functions in nvector.objects.py
         Replaced f-string in the __repr__ method of the _Common class in
         nvector.objects.py with format in order to work on python version 3.5
         and below. 
      * Made nvector.plot.py more robust.
      * Removed rtol parameter from the on_greatcircle function. See issue #12 for a discussion.
      * Added nvector solution to the GeoPoint.displace method.
      * Updated docs/conf.py
      * Updated README.rst and LICENSE.txt
      * Replaced import unittest with import pytest in test_frames.py
      * Fixed issue #10: Inconsistent return types in GeoPath.track_distance:
         - GeoPath, GeoPoint, Nvector and ECEFvector and Pvector now return
           scalars for the case where the input is not actually arrays of points
           but just single objects.
      * Added extra tests for issue #10 and updated old tests and the examples in the help headers.
      * Vectorized FrameE.inverse and FrameE.direct methods.
      * Extended deg and rad functions in _core.py.
      * Vectorized GeoPoint.distance_and_azimuth
      * Made import of cartopy in nvector.plot more robust.
      * Updated test_Ex10_cross_track_distance
      * Updated sonar-project.properties
      * Replaced deprecated sonar.XXXX.reportPath with sonar.XXXX.reportPaths
      * Simplified nvector/_core.__doc__
      * Updated .travis.yml
      * Changed the definition of sonar addon
      * Added CC_TEST_REPORTER_ID to .travis.yml
      * Added python 3.8 to the CI testing.
      * Changed so that setup.py is python 2.7 compatible again.
      * Updated build_package.py
      * Renamed CHANGES.rst to CHANGELOG.rst
      * Updated setup.cfg and setup.py
      * Added license.py
      * Updated build_package.py
      * Removed conda-build from .travis.yml
      * Attempt to get travis to run the tests again....
      * API change: replaced "python setup.py doctests" with "python setup.py doctest"
      * Added doctest example to nvector._core._atleast_3d Made xyz2R and zyx2R code simpler.
      * Replaced deprecated Nvector.mean_horizontal_position with  Nvector.mean in test_frames.py
      * Added mdot to __all__ in nvector/_core.py and in documentation summary.
      * Sorted the the documentation summary by function name in nvector.rst
      * Removed --pyargs nvector --doctest-modules --pep8 from addopts section in setup.cfg
      * Updated documentation and added missing documentation.


Version 0.7.4, June 4, 2019
===========================

Per A Brodtkorb (2):
      * Fixed PyPi badge and added downloads badge in nvector/_info.py and README.rst
      * Removed obsolete and wrong badges from docs/index.rst


Version 0.7.3, June 4, 2019
===========================

Per A Brodtkorb (6):
      * Renamed LICENSE.txt and THANKS.txt to LICENSE.rst and THANKS.rst
      * Updated README.rst and nvector/_info.py
      * Fixed issue 7# incorrect test for test_n_E_and_wa2R_EL.
      * Removed coveralls test coverage report.
      * Replaced coverage badge from coveralls to codecov.
      * Updated code-climate reporter.
      * Simplified duplicated code in nvector._core.
      * Added tests/__init__.py
      * Added "--pyargs nvector" to pytest options in setup.cfg
      * Exclude build_package.py from distribution in MANIFEST.in
      * Replaced health_img from landscape to codeclimate.
      * Updated travis to explicitly install pytest-cov and pytest-pep8
      * Removed dependence on pyscaffold
      * Added MANIFEST.in
      * Renamed set_package_version.py to build_package.py


Version 0.7.0, June 2, 2019
============================

Gary van der Merwe (1):
      * Add interpolate to __all__ so that it can be imported

Per A Brodtkorb (26):
      * Updated long_description in setup.cfg
      * Replaced deprecated sphinx.ext.pngmath with sphinx.ext.imgmath
      * Added imgmath to requirements for building the docs.
      * Fixing shallow clone warning.
      * Replaced property 'sonar.python.coverage.itReportPath' with
         'sonar.python.coverage.reportPaths' instead, because it is has been removed.
      * Drop python 3.4 support
      * Added python 3.7 support
      * Fixed a bug: Mixed scalars and np.array([1]) values don't work with np.rad2deg function.
      * Added ETRS ELLIPSOID in _core.py Added ED50 as alias for International
         (Hayford)/European Datum in _core.py Added sad69 as alias for South American 1969 in _core.py
      * Simplified docstring for nv.test
      * Generalized the setup.py.
      * Replaced aliases with the correct names in setup.cfg.


Version 0.6.0, December 9, 2018
===============================

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
     * Added great_circle_normal, cross_track_distance
     * Renamed intersection to intersect (Intersection is deprecated.)
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
      * Renamed distance_rad_bearing_rad2point to n_EA_E_distance_and_azimuth2n_EB_E
      * Renamed azimuth to n_EA_E_and_n_EB_E2azimuth     
      * Added tests for R2xyz as well as R2zyx
      * Removed backward compatibility     
      * Added test_n_E_and_wa2R_EL
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
