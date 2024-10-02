=========
Changelog
=========

Version 1.0.0, October 2, 2024
==============================
Per A Brodtkorb (63):
      
      * Added pytest-ruff and ruff as test dependencies to pyproject.toml
      * added sphinx_build_latex script to pyproject.toml
      * Replaced all single quotes (') with double quotes (").  
      * Added pretty print __str__ method to _Common in objects.py
      * Removed conversion of an array with ndim > 0 to a scalar in plot.py since it is deprecated, and will error in future.
      * Replaced dependency on geographiclib with karney.py, which now is a separate package.
      * Made matplotlib and cartopy optional dependency
      * Added dm2degrees and degrees2dm in util.py
      * Adopted many of the changes Matt Hogan did to the nvector fork called "envector":  
      
         * Updated docstrings to adopt the Napoleon docstring standard.  
         * Removed numpydoc from the documentation requirements as it is not used.  
         * Added type-hints to most functions and methods  
         * Reduced matrix of supported Python versions as to reduce burden of testing  
         * Added local testing support using nox.
         * Replaced testing on travis and appveyor with github.workflows/python-package
         * Added CODE_OF_CONDUCT.md, CONTRIBUTING.rst and workflows/python-package.yml.
         * Removed THANKS.rst

      * Added example to great_circle_distance_rad
      * Replaced numpy.finfo(float).tiny with numpy.finfo(float).smallest_normal if version > 1.22 in util.py
      * Added main function to plot.py
      * Made _init_earth_plotter more robust in nvector.plot.py.
      * Made test_prolate15_direct more forgiving...
      * Removed unused rsum from util.py
      * Fixed doctest so they don't crash on travis: Replaced "# doctest + SKIP" with "# doctest: +SKIP" in docstrings.
      * Added hypothesis to tests_require in setup.py
      * Updated badges in README.rst and _images.py
      * Simplified duplicated test-values in test_R2zyx_zyx2R_roundtrip and test_R2xyz_xyz2R_roundtrip.
      * Updated example 5 in _examples.py so it works again. 
      * Fixed a bug in geodesic_distance
      * Moved test_geo_solve6, test_geo_solve9,  test_geo_solve10 and test_geo_solve11 into test_inverse_cornercases
      * Fixed a bug in test_nan_propagation
      * Moved test_geo_solve29, test_geo_solve0 and test_geo_solve1 into WGS84_TESTCASES
      * Removed obsolete long_unroll argument from GeoPoint.distance_and_azimuth and FrameE.inverse
      * Fixed karney.py so that all the tests work.
      * Added geodesic_reckon and improved geodesic_distance in core.py 
      * Added karney.py
      * Fixed doctests in lat_lon2n_E in core.py 
      * Added geodesic_distance function to core.py 
      * Added doctest example to GeoPoint.distance_and_azimuth method in objects.py
      * Added allclose to __all__ in nvector.util.py and replaced all examples using np.allclose with nv.allclose 
      * Added doctest example to lat_lon2n_E function. 
      * Made lat_lon2n_E more general by allowing to broadcast the input. 
      * Added test_util.py module. 
      * Made nthroot function more robust against zero-division. 
      * Made unit function more robust against overflow. 
      * Made get_ellipsoid more forgiving on name argument
      * Added NGO1948 and EUREF89 options to get_ellipsoid.
      * Added course_over_ground, n_EA_E_and_n_EB_E2p_AB_N and n_EA_E_and_p_AB_N2n_EB_E to core.py 
      * Added course_over_ground method to Nvector class in objects.py
      * Added  n_EA_E_and_n_EB_E2p_AB_N and n_EA_E_and_p_AB_N2n_EB_E functions to core.py 
      * Added content to topics/nvector.rst
      * Fixes issue #15: KeyError: 'point' in __repr__ of Pvector
      * Added test_nvector_with_vectors.py in order to check vectorized code work correctly. 
      * Refactored great_circle_distance_rad function from great_circle_distance function
      * Removed unused code from _info.py, _info_functional.py, core.py, plot.py
      * Added tests to check if ``zyx2R(*R2zyx(r_matrix)) == r_matrix`` and ``xyz2R(*R2xyz(r_matrix)) == r_matrix`` 
      * Corrected docstring of mdot in nvector.util.py



Version 0.7.7, June 3, 2021
================================
Per A Brodtkorb (27):
      * Added cartopy and matplotlib to requirements.txt
      * Updated appveyor.yml, setup.cfg and setup.py
      * Updated .gitignore to ignore .pytest_cache
      * Corrected failing doctests in objects.py
      * Updated version in _installation.py
      * Updated failing docstrings for python 2.7 in objects.py.
      * Added '# doctest: SKIP' to all plt.show() in order to avoid the doctests hangs on the testserver.
      * Fixed a bug  in _info_functional.py
      * Updated pycodestyle exlude section in setup.cfg Prettified _examples.py, _examples_object_oriented.py and core.py
      * Updated pycodestyle ignore section in setup.cfg
      * Added doctest option to setup.cfg
      * Removed print statements in test_objects.py
      * Return "NotImplemented" instead of raising "NotImplementedError" in Nvector._mul__ and Nvector.__div__ in objects.py
      * Fixed .travis.yml so that he file paths in coverage.xml is discoverable
         under the sonar.sources folder. The problem is that SonarQube is
         analysing the checked-out source code (in src/nvector) but the actual
         unit tests and coverage.py is run against the installed code (in
         build/lib/nvector). Thus the absolute files paths to the installed code
         in the generated coverage.xml were causing Sonar to show no coverage.
         The workaround was to use sed in the pipeline to replace every path to
         build/lib/nvector with src/nvector in coverage.xml.
      * Fixed a bug: Identical expressions should not be used on both sides of a binary operator in test:objects.py.
      * Updated solutions to example 9
      * Added greatcircle method to GeoPoint.distance_and_azimuth in objects.py
      * Added _base_angle function that makes sure an angle is between -pi and pi. 
      * Added test_direct_and_inverse in test_objects.py
      * Added interp_nvectors to docs/reference/nvector_summary.rst
      * Added vectorized interpolation routines: interp_nvectors function to core.py and Nvector.interpolate to objects.py.
      * Put try except around code in use_docstring to avoid attribute '__doc__'
         of 'type' objects is not writable errors for  python2. 
      * Added interp_nvectors 
      * Reorganized _displace_great_circle 
      * Added check that depths also are equal on in _on_ellipsoid_path and in _on_great_circle_path
      * Refactored code from use_docstring_from function into the use_docstring
         function in _common.py 
      * Simplified the adding of examples to the docstrings of functions and classes in core.py and objects.py.

Version 0.7.6, December 18, 2020
================================

Per A Brodtkorb (30):
      * Renamed _core.py to core.py 
      * Removed the module index from the appendix because it was incomplete. 
      * Removed nvector.tests package from the reference chapter. 
      * Added indent function to _common.py to avoid failure on python 2.7.
      * Moved isclose, allclose and array_to_list_dict from objects.py to util.py
      * Moved the following function from test_nvector.py to test_rotation.py:
          - test_n_E_and_wa2R_EL, test_R2zxy, test_R2zxy_x90, test_R2zxy_y90
          - test_R2zxy_z90, test_R2zxy_0, test_R2xyz test_R2xyz_with_vectors 
      * Replaced assert_array_almost_equal with assert_allclose in test_objects.py
      * Renamed test_frames.py to test_objects.py
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
