#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for nvector.

    This file was generated with PyScaffold 2.4.4, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/

Usage:
Run all tests:
  python setup.py test

  python setup.py doctests

Build documentation

  python setup.py docs

Install
  python setup.py install [, --prefix=$PREFIX]

Build

  python setup.py bdist_wininst

  python setup.py bdist_wheel --universal

  python setup.py sdist

PyPi upload:
  git pull origin
  git tag v0.5.1 master
  git shortlog v0.4.1..v0.5.1 > log.txt  # update Changes.rst
  git commit
  git tag v0.5.1 master
Delete the build, dist, and nvector.egg-info folder in your root directory.
  python setup.py sdist
  python setup.py bdist_wheel --universal
  python setup.py egg_info
  git push --tags
  twine -p PASSWORD upload dist/*


"""

import sys
from setuptools import setup, find_packages


def _get_version_from_pkg():
    import pkg_resources
    try:
        version = pkg_resources.get_distribution("nvector").version
        with open("__conda_version__.txt", "w") as fid:
            fid.write(version)
    except pkg_resources.DistributionNotFound:
        version = 'unknown'
    return version


def _get_version_from_git():
    import subprocess
    try:
        version = subprocess.check_output("git describe --tags").decode('utf-8')
        version = version.lstrip('v').strip()
    except Exception: # subprocess.CalledProcessError:
        version = 'unknown'
    parts = version.split('-')
    if len(parts) == 1:
        version = parts[0]
    elif len(parts) == 3:
        tag, revision, sha = parts
        version = '{}.post{:03d}+{}'.format(tag, int(revision), sha)
    else:
        version = 'unknown'
    return version


def get_version():
    version = _get_version_from_git()
    if version == 'unknown':
        return _get_version_from_pkg()
    return version


def update_version_in_package(version):
    import re
    if version != 'unknown':
        with open("./src/nvector/__init__.py", "r") as fid:
            text = fid.read()

        new_text = re.sub(r"__version__ = ['\"]([^'\"]*)['\"]",
                          '__version__ = "{}"'.format(version),
                          text, re.M)

        with open("./src/nvector/__init__.py", "w") as fid:
            fid.write(new_text)


def setup_package():
    version = get_version()
    update_version_in_package(version)
    print("Version: {}".format(version))

    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx', 'numpydoc',
              'sphinx_rtd_theme>=0.1.7'] if needs_sphinx else []
    setup(setup_requires=['pyscaffold==2.5.11'] + sphinx,
          package_dir = {'': 'src'},
          include_package_data=True,
          packages=find_packages(where=r'./src'),
          tests_require=['pytest_cov', 'pytest'],
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
