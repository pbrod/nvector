"""
Created on 7. des. 2018

@author: pab
"""
from __future__ import absolute_import
import sys
import re


def update_version_in_package(version):
    if version != 'unknown':
        print("Version: {}".format(version))
        with open("./src/nvector/__init__.py", "r") as fid:
            text = fid.read()

        new_text = re.sub(r"__version__ = ['\"]([^'\"]*)['\"]",
                          '__version__ = "{}"'.format(version),
                          text, re.M)  # @UndefinedVariable

        with open("./src/nvector/__init__.py", "w") as fid:
            fid.write(new_text)


def main(argv):
    """Main entry point into the converter script"""

    if len(argv) != 2:
        print("Usage  python set_version_in_package <version>")
        sys.exit(0)

    version = argv[1]
    update_version_in_package(version)


if __name__ == "__main__":
    main(sys.argv)
    # main(('', '0.6.0rc'))
