"""
Created on 7. des. 2018

@author: pab
"""
import os
import re
import shutil
import subprocess

import click


ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'nvector'


def remove_previous_build():
    egginfo_path = os.path.join('src', '{}.egg-info'.format(PACKAGE_NAME))
    docs_folder = os.path.join(ROOT, 'docs', '_build')

    for dirname in ['dist', 'build', egginfo_path, docs_folder]:
        path = os.path.join(ROOT, dirname)
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument("version")
def build_main(version):
    """Build and update {} version, documentation and package.

    The script remove the previous built binaries and generated documentation
    before it generate the documentation and build the binaries and finally
    check the built binaries.
    """.format(PACKAGE_NAME)
    remove_previous_build()
    set_package(version)

    for cmd in ['docs', 'sdist', 'bdist_wheel', 'egg_info']:
        try:
            subprocess.call(["python", "setup.py", cmd])
        except Exception as error:  # subprocess.CalledProcessError:
            print('{}: {}'.format(cmd, str(error)))
    try:
        subprocess.call(["twine", "check", "dist/*"])
    except Exception as error:  # subprocess.CalledProcessError:
        print("Twine: ", str(error))


def set_package(version):
    """Set version of {} package""".format(PACKAGE_NAME)

    if version:
        filename = "{}/src/{}/__init__.py".format(ROOT, PACKAGE_NAME)
        print("Version: {}".format(version))
        with open(filename, "r") as fid:
            text = fid.read()

        new_text = re.sub(r"__version__ = ['\"]([^'\"]*)['\"]",
                          '__version__ = "{}"'.format(version),
                          text, re.M)  # @UndefinedVariable

        with open(filename, "w") as fid:
            fid.write(new_text)


if __name__ == "__main__":
    build_main()
