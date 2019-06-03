#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for nvector.


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
  git shortlog v0.6.0..HEAD -w80 --format="* %s" --reverse > log.txt  # update Changes.rst
  python build_package.py 0.7.0rc0
  git commit
  git tag v0.7.0rc0 master
  git push --tags
  twine check dist/*   # check
  twine upload dist/*

"""
import os
import re
import sys
from setuptools import setup, Command
ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'nvector'


def read(filename):
    with open(filename, 'r') as file_handle:
        return file_handle.read()


def get_version():
    filename = os.path.join(ROOT, "src", PACKAGE_NAME, "__init__.py")
    text = read(filename)
    versions = re.findall(r"__version__ = ['\"]([^'\"]*)['\"]",
                          text, re.M)  # @UndefinedVariable
    return versions[0]


class Doctest(Command):
    description = 'Run doctests with Sphinx'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.application import Sphinx
        sph = Sphinx('./docs',  # source directory
                     './docs',  # directory containing conf.py
                     './docs/_build',  # output directory
                     './docs/_build/doctrees',  # doctree directory
                     'doctest')  # finally, specify the doctest builder
        sph.build()


def setup_package():
    version = get_version()
    print("Version: {}".format(version))

    sphinx_requires = ['sphinx>=1.3.1']
    needs_sphinx = {'build_sphinx'}.intersection(sys.argv)
    sphinx = ['sphinx', 'numpydoc', 'imgmath',
              'sphinx_rtd_theme>=0.1.7'] if needs_sphinx else []
    setup(setup_requires=["pytest-runner"] + sphinx,
          version=version,
          cmdclass={'doctests': Doctest
                    },
          extras_require={'build_sphinx': sphinx_requires,
                          },
          )


if __name__ == "__main__":
    setup_package()
