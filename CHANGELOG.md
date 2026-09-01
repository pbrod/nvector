# Changelog

## [1.2.0] - 2026-09-01


### 🐛 Bug Fixes

- *(__init__)* Explicit re-exports and `__all__` definition to resolve Ruff F401
- *(nvector.test)* Remove unsupported `plugins` argument
- *(nvector.testing)* Add missing `sys` import
- *(nvector.__init__)* Remove unused `Any` import


### 📦 Dependencies

- Update lockfiles


### ⚙️ Maintenance

- Update CI configuration
- General project maintenance updates
- *(cliff)* Improve changelog generation and commit grouping
- *(pyproject)* Enhance project metadata, tooling, and release documentation
  - Add keywords and Python classifiers
  - Standardize project URLs
  - Enable Ruff pyupgrade checks
  - Document release and maintenance workflow


### ♻️ Refactoring

- *(test)* Run pytest in a subprocess
- *(__init__.py)* Replace wildcard imports with explicit imports
- Refactor dynamic docstring assembly for improved safety and readability
- *(_typing)* Replace `Union` with PEP 604 union syntax
  - Use `X | Y` type annotations
  - Remove legacy `typing.Union` usage
  - Align type hints with Python 3.10+ requirements
  - Satisfy Ruff UP007 checks


### 📚 Documentation

- *(pyproject.toml)* Update build process documentation
- *(changelog)* Normalize historical release history
- *(changelog)* Add git-cliff configuration


### 🎨 Styling

- *(pyproject.toml)* Fix typos
- General style cleanups


### 🧪 Testing

- Add tests for `testing.py`
- Fix mypy errors in `test_testing.py`
- Remove obsolete `test_test_forwards_plugins`
- Remove obsolete `pytest.main` tests


### 📦 Build System

- *(deps)* Update `pyproject.toml` syntax
- Update release workflow scripts and automation


### 🏗️ CI/CD

- Fix caching paths and coverage generation in CI workflow
- Fix Python 3.10 Windows wheel-install test
- Add Python 3.15 RC testing and update lock files
- Replace Python 3.15 with 3.15.0-rc.1 in CI
- Update `actions/checkout` to v5 in release workflow


## [1.1.0] - 2026-08-04

### 🐛 Bug Fixes

- Fixed the maintainability badge.
- Fixed imports for `test_docstrings` and docstring examples.

### 🚜 Refactor

- Improved type annotations and type checking across the code base.
- Moved `test` and `test_docstrings` utilities into `testing.py`.
- Simplified `pyproject.toml` and refreshed lock files.

### 📚 Documentation

- Added release workflow documentation (`RELEASE.md`).
- Improved release, deployment, and contributor documentation.
- Updated project documentation and quality badges.
- Added `update_license.py` and `update_readme.py` utilities to automate generation of project documentation files from source modules.

### 🧪 Testing

- Added explicit Ruff style checks to CI.
- Added explicit MyPy type checks to CI.
- Added support for testing against Python 3.15.
- Added a `test` PDM script and documented pytest markers.

### ⚙️ CI/CD

- Simplified the CI workflow configuration.
- Improved lock-file handling in CI.
- Added package build verification and wheel installation checks.
- Added PyPI release validation to prevent publishing existing versions.
- Modernized GitHub Actions CI and PyPI release workflows.
- Removed obsolete documentation build configuration.


## [1.0.6] - 2025-12-02

### 🐛 Bug Fixes

-  Added missing myst-parser to docs/requirements.txt.
-  Corrected the pypi-publish job's if condition  to ensure the it triggers correctly whenever a tag starting with v is pushed, including when using git push --tags --force.
### 📚 Documentation

- Updated link to latest version of the documentation on readthedocs
- Updated CONTRIBUTING.rst

### ⚙️ Miscellaneous Tasks

- Send coverage report to codecov again.
- Deleted obsolete platform specific pdm-lock-files.
- Deleted obsolete sonar-project.properties.
- Added --mypy to pytest.ini_options.
- Only publish package if the version does not exist on pypi.
- Split github actions file python-package.yml into ci-test.yml and release.yml.

## [1.0.3] - 2025-12-01

### 🐛 Bug Fixes

- Continous integration now works for python 3.10-3.14

### 📚 Documentation

- *(pyproject.toml)* Replace use of python-semantic-release with pdm-bump and git-cliff.
- Updated test badge in README.rst

### 🧪 Testing

- Fix incorrect use platform-specific steps ci and coverage jobs

### ⚙️ Miscellaneous Tasks

- Change project short description
- Bumped minimum required Python version up to 3.10.
- Add script to gen macos lock file
- Add macos lock file
- Update .readthedocs.yml
- Simplified python-package.yml
- Disabled sending coverage report to codeclimate, codecov and SonarCloud
- Remove python version in lock files
- Remove release1, pypi-publish and test-pypi-publish jobs from workflows/python-package.yml


## [1.0.2] - 2024-10-04

### 📚 Documentation

- Updated test badge and quality badge in README.rst

### ⚙️ Miscellaneous Tasks

- Add pdm.lock and cliff.toml
- Update release.yml and python-package.yml
- Send coverage report to codeclimate and codecov
- Only send code coverage reports for changes made to the master branch
- Add sonar-project.properties and update link to test badge
- Update SonarCloud analysis in python-package.yml
- Update .readthedocs.yml

## [1.0.1] - 2024-10-03

### 📚 Documentation

- Prepare changelog for automatic updates using semantic versioning

### ⚙️ Miscellaneous Tasks

- *(pyproject.toml)* Replace use of python-semantic-release with pdm-bump and git-cliff.

### ◀️ Revert

- Remove release1, pypi-publish and test-pypi-publish jobs from workflows/python-package.yml

## [1.0.0] - 2024-10-02

### ✨ Features

- Added `dm2degrees` and `degrees2dm` to `util.py`.
- Added `great_circle_distance_rad`.
- Added `geodesic_distance` and `geodesic_reckon`.
- Added `karney.py` and replaced the dependency on GeographicLib.
- Added examples for geodesic and great-circle calculations.
- Added `course_over_ground`.
- Added `n_EA_E_and_n_EB_E2p_AB_N` and `n_EA_E_and_p_AB_N2n_EB_E`.
- Added support for `NGO1948` and `EUREF89` in `get_ellipsoid`.
- Added `Nvector.course_over_ground`.

### 🐛 Bug Fixes

- Fixed `geodesic_distance`.
- Fixed `karney.py` compatibility issues.
- Fixed doctests in `lat_lon2n_E`.
- Fixed `test_nan_propagation`.
- Updated example 5 in `_examples.py`.
- Made `unit` more robust against overflow.
- Made `nthroot` more robust against division by zero.
- Made `lat_lon2n_E` support broadcasting.
- Made `_init_earth_plotter` more robust.
- Removed deprecated NumPy scalar conversions.
- Updated deprecated uses of `numpy.finfo(...).tiny`.

### ♻️ Refactoring

- Replaced all single quotes with double quotes.
- Refactored `great_circle_distance_rad` from `great_circle_distance`.
- Removed unused code from several modules.
- Removed obsolete arguments from geodesic APIs.
- Simplified duplicated test data and examples.
- Added pretty-printing support to `_Common.__str__`.

### 📚 Documentation

- Updated docstrings to the Napoleon style.
- Removed unused `numpydoc` dependency.
- Added examples for geodesic and great-circle functions.
- Expanded documentation and reference material.
- Added content to `topics/nvector.rst`.
- Updated badges and project documentation.

### 🧪 Testing

- Added extensive type hints throughout the code base.
- Added local testing support with nox.
- Added `pytest-ruff` and Ruff.
- Added `hypothesis` testing support.
- Expanded vectorization and geodesic tests.
- Updated numerous doctests.

### ⚙️ Maintenance

- Reduced supported Python-version matrix.
- Replaced Travis CI and AppVeyor with GitHub Actions.
- Added `CODE_OF_CONDUCT.md` and `CONTRIBUTING.rst`.
- Made Cartopy and Matplotlib optional dependencies.
- Replaced GeographicLib dependency with `karney.py`.
- Removed `THANKS.rst`.


## [0.7.7] - 2021-06-03

### ✨ Features

- Added vectorized interpolation support with `interp_nvectors`.
- Added `GeoPoint.distance_and_azimuth(..., method="greatcircle")`.
- Added `_base_angle` utility for angle normalization.
- Added interpolation support to `Nvector`.

### 🐛 Bug Fixes

- Corrected failing doctests in `objects.py`.
- Fixed bugs in `_info_functional.py`.
- Fixed SonarQube coverage path handling in CI.
- Corrected solutions for example 9.
- Improved robustness of docstring handling on Python 2.7.

### ♻️ Refactoring

- Refactored docstring generation utilities.
- Reorganized `_displace_great_circle`.
- Simplified generation of docstring examples.
- Improved internal path checking logic.

### 📚 Documentation

- Added `interp_nvectors` to reference documentation.
- Updated installation and usage documentation.
- Updated examples and object-oriented tutorials.
- Improved generated API documentation.

### 🧪 Testing

- Added `test_direct_and_inverse`.
- Added additional interpolation tests.
- Updated doctest configuration.

### ⚙️ Maintenance

- Added Cartopy and Matplotlib dependencies.
- Updated setup configuration and CI files.
- Updated version information and ignored cache files.


## [0.7.6] - 2020-12-18

### ✨ Features

- Added `logo.png` and Nvector branding assets.
- Added comprehensive tutorial structure and documentation sections.
- Added rotation matrix documentation and examples.
- Added utility functions for documentation generation.

### ♻️ Refactoring

- Renamed `_core.py` to `core.py`.
- Renamed `select_ellipsoid` to `get_ellipsoid`.
- Moved rotation-matrix functions into a dedicated `rotation` module.
- Moved common utility functions into `util.py`.
- Renamed requirements and documentation files to improve consistency.

### 📚 Documentation

- Reorganized documentation into:
  - Introduction
  - Tutorials
  - How-to guides
  - Reference
  - Appendix
- Added Read the Docs configuration.
- Added BibTeX references and bibliography support.
- Expanded rotational mathematics documentation.
- Added installation guide and documentation assets.

### 🧪 Testing

- Moved rotation-related tests to dedicated test modules.
- Improved correctness of documentation examples.

### ⚙️ Maintenance

- Updated build and release tooling.
- Removed obsolete documentation artifacts and image files.
- Updated acknowledgements and references.


## [0.7.5] - 2020-12-12

### ✨ Features

- Added `GeoPoint.displace`.
- Added vectorized implementations of:
  - `GeoPoint.distance_and_azimuth`
  - `FrameE.direct`
  - `FrameE.inverse`
- Added `allclose`, `array_to_list_dict`, and `isclose`.
- Added additional geodesic and path functionality.

### 🐛 Bug Fixes

- Fixed issue #10 regarding inconsistent scalar and array return types.
- Fixed failing doctests.
- Improved robustness of Cartopy imports.
- Updated deprecated Sonar configuration.

### ♻️ Refactoring

- Simplified internal documentation generation.
- Simplified rotation matrix code.
- Replaced deprecated methods and APIs.

### 📚 Documentation

- Added extensive doctest examples.
- Improved API reference documentation.
- Updated README and project documentation.

### 🧪 Testing

- Added regression tests for issue #10.
- Added Python 3.8 CI coverage.
- Expanded geodesic and path test coverage.

### ⚙️ Maintenance

- Renamed `CHANGES.rst` to `CHANGELOG.rst`.
- Updated setup configuration and packaging.
- Removed obsolete CI configuration.


## [0.7.4] - 2019-06-04

### 🐛 Bug Fixes

- Fixed PyPI badge links.
- Removed obsolete and incorrect badges from the documentation site.

### 📚 Documentation

- Updated project badges in:
  - `README.rst`
  - `docs/index.rst`
  - `nvector/_info.py`

### ⚙️ Maintenance

- Refreshed project metadata and status badges to reflect the current project infrastructure.


## [0.7.3] - 2019-06-04

### 🐛 Bug Fixes

- Fixed issue #7 in `test_n_E_and_wa2R_EL`.
- Updated badges and package metadata.
- Improved duplicated code in `nvector._core`.

### 🧪 Testing

- Added `tests/__init__.py`.
- Added `--pyargs nvector` to pytest configuration.
- Updated Travis CI configuration.

### ⚙️ Maintenance

- Renamed `LICENSE.txt` and `THANKS.txt` to reStructuredText files.
- Added `MANIFEST.in`.
- Renamed `set_package_version.py` to `build_package.py`.
- Removed dependency on PyScaffold.
- Replaced Coveralls with Codecov.
- Updated Code Climate integration.
- Added explicit installation of `pytest-cov` and `pytest-pep8` in CI.

### 📚 Documentation

- Updated `README.rst`.
- Updated package information displayed in `_info.py`.


## [0.7.0] - 2019-06-02

### ✨ Features

- Added `interpolate` to the public API.
- Added support for the ETRS ellipsoid.
- Added aliases for ED50 and SAD69 ellipsoids.
- Added support for Python 3.7.

### 🐛 Bug Fixes

- Fixed handling of mixed scalar and array input in angle conversions.
- Replaced deprecated SonarQube configuration options.

### 📚 Documentation

- Added support for Sphinx `imgmath`.
- Simplified documentation for `nv.test`.
- Updated package metadata and long description.

### ⚙️ Maintenance

- Dropped support for Python 3.4.
- Generalized `setup.py`.
- Replaced deprecated aliases in package configuration.
- Updated CI and SonarCloud integration.


## [0.6.0] - 2018-12-09

### ✨ Features

- Added `delta_L`.
- Added vectorized frame support for multiple positions.
- Added additional displacement and path test cases.

### 🐛 Bug Fixes

- Avoid division-by-zero in `unit`.
- Fixed multiple doctest failures.
- Improved compatibility with Cartopy.

### ♻️ Refactoring

- Moved package into `src/nvector`.
- Refactored duplicated code.
- Simplified object-oriented examples.

### 📚 Documentation

- Updated documentation and installation instructions.
- Updated API documentation and examples.

### 🧪 Testing

- Added path and displacement test cases.
- Expanded frame and geometry tests.

### ⚙️ Maintenance

- Added SonarCloud and CodeClimate integration.
- Added `.pylintrc`.
- Updated dependencies and CI configuration.


## [0.5.2] - 2017-03-07

### 🐛 Bug Fixes

- Fixed failing tests in `tests/test_frames.py`.
- Updated cross-track distance tests.
- Updated documentation links.

### 📚 Documentation

- Updated `README.rst`.
- Added example 10 image.
- Updated links to online documentation.

### 🧪 Testing

- Updated `tox.ini`.
- Updated Travis CI configuration.
- Continued PEP 8 cleanup.

### ⚙️ Maintenance

- Updated project configuration files.
- Updated package metadata and build settings.


## [0.5.1] - 2017-03-05

### ✨ Features

- Added:
  - `GeoPath.on_path`
  - `GeoPath.on_great_circle`
  - `GeoPath.closest_point_on_path`
  - `GeoPath.closest_point_on_great_circle`
  - `great_circle_normal`
  - `cross_track_distance`

### 🐛 Bug Fixes

- Improved numerical accuracy for cross-track distance calculations.
- Made frame equality comparisons more robust.
- Replaced zero n-vectors with NaN representations.

### ♻️ Refactoring

- Removed duplicated code.
- Simplified rotation conversions.
- Moved docstring helpers into `_common.py`.

### 📚 Documentation

- Updated README and installation documentation.
- Added example 10 and supporting figures.
- Expanded usage examples for path calculations.

### 🧪 Testing

- Added extensive tests for great-circle functionality.
- Updated and improved existing test coverage.

### ⚙️ Maintenance

- Added Code Climate configuration.
- Updated project metadata and classifiers.


## [0.4.1] - 2016-01-19

### ✨ Features

- Added `GeoPath.interpolate`.
- Added interpolation example 6.
- Added figures to documentation examples.

### ♻️ Refactoring

- Extracted `_check_frames`.
- Extracted `_default_frame`.
- Consolidated example generation in `_examples.py`.

### 📚 Documentation

- Expanded FrameB, FrameE, FrameL and FrameN documentation.
- Added links to GeographicLib and the n-vector MATLAB toolbox.
- Improved examples and tutorial content.

### ⚙️ Maintenance

- Renamed `info.py` to `_info.py`.
- Updated coverage and Sphinx configuration.
- Updated package metadata.


## [0.1.3] - 2016-01-01

### ✨ Features

- Added tests for `R2xyz` and `R2zyx`.
- Added `test_n_E_and_wa2R_EL`.
- Added `tox.ini`.
- Added Travis CI and Landscape configuration.

### 🐛 Bug Fixes

- Fixed setup configuration issues.
- Fixed Travis CI configuration.

### ♻️ Refactoring

- Renamed:
  - `distance_rad_bearing_rad2point` → `n_EA_E_distance_and_azimuth2n_EB_E`
  - `azimuth` → `n_EA_E_and_n_EB_E2azimuth`
- Refactored tests and project structure.
- Removed dependence on `navigator.py`.
- Removed backward compatibility code.
- Moved tests into `nvector/tests`.

### 📚 Documentation

- Updated `README.rst`.
- Updated examples and changelog.
- Improved project documentation.

### 🧪 Testing

- Expanded rotation-matrix test coverage.
- Updated and reorganized tests.
- Disabled a few unstable Python 3 tests pending fixes.

### ⚙️ Maintenance

- Deleted obsolete files and licenses.
- Updated coverage configuration.
- Added CI support and project tooling.


## [0.1.1] - 2016-01-01

### ✨ Features

- Initial Python release of nvector.
- Added object-oriented interface.
- Added geodesic functionality.
- Added geodesic test suite.

### 🧪 Testing

- Added tests for the object-oriented API.
- Added geodesic regression tests.

### ⚙️ Maintenance

- Initial project structure and packaging.
