Contributing
============

Bug reports, feature suggestions, and other contributions are greatly appreciated!

Short version
-------------

* Submit bug reports and feature requests at 
  [GitHub](https://github.com/pbrod/nvector/issues)

* Make pull requests to the ``develop`` branch.

Bug reports
-----------

When [reporting a bug](https://github.com/pbrod/nvector/issues) please
include:

* Your operating system name and version

* Any details about your local setup that might be helpful in troubleshooting

* Detailed steps to reproduce the bug


Feature requests and feedback
-----------------------------

The best way to send feedback is to file an issue at
[GitHub](https://github.com/pbrod/nvector/issues).

If you are proposing a feature:

* Explain in detail how it would work.

* Keep the scope as narrow as possible, to make it easier to implement.

* Remember that this is a volunteer-driven project, and that code contributions
  are welcome :)


Development Prerequisites
-------------------------

To set up `nvector` on your local host for development, you only need the Git application. You can download Git using
your distributions preferred method (dnf, yum, apt-get, brew) or using GitHub Desktop. 

The following are optional, but recommended.

* The CPython interpreter version 3.12
   * You can install it using official binaries, pyenv, or any Anaconda-like distribution 
* The [PDM](https://pdm-project.org/latest/) application for locking, building, and testing.
* Dedicated IDE like PyCharm, Spyder, or [VSCodium](https://vscodium.com/) to name a few. These allow more flexible git
  control.


Development Steps
-----------------

1. [Fork nvector on GitHub](https://github.com/pbrod/nvector)

2. Clone your fork locally. Using the command line would be:

   ```shell
   git clone git@github.com/<USER>/nvector.git
   ```
   
3. Create a branch for local development. Using the command line would be:

   ```shell
   git checkout -b name-of-your-bugfix-or-feature
   ```
   Now you can make changes and commit them.
  
4. When you're done making changes, run all the checks to ensure that nothing
   is broken on your local system. To run tests locally, you should install a PDM virtual environment 
   first

   ```shell
   pdm use -i 3.12  # Automatically searches for Python3.12
   pdm use /path/to/python3.12  # Instead specify path to Python3.12
   pdm install -L pdm_<PLATFORM>.lock  # Replace <PLATFORM> with linux, windows
   ```
    
    Now you can run tests

    ```shell
    pdm run pytest 
    ```
   
   If you have multiple Pythons installed to your path (3.9+), then you can use nox

   ```shell
   pdm run nox
   ```
   
5. Run any linting or style compliance program. Currently `nvector` does not have any, but a suggestion would be 
appreciated.

6. Update/add documentation (in ``docs``), if relevant.
   
7. Add your name to the ``AUTHORS.rst`` file as an author.

8. Commit your changes. Using the command line would be:

   ```shell
   git add <FILE1> <FILE2> ... <FILEN>
   git commit -m "TAG: Brief description. Longer description later."
   ```
   
   The `nvector` package does not follow any development workflows at this time, but one should be adopted like NumPy
   workflow with tags.

9. Once you are happy with the local changes, push to GitHub:

   ```
   git push origin name-of-your-bugfix-or-feature
   ```
   
   Note that each push will trigger the Continuous Integration workflow. Check the ``Actions`` tab on your fork 
   repository home like github.com/<USER>/nvector/actions

10. Submit a pull request through the GitHub website. Pull requests should be
    made to the ``develop`` branch (subject to change).  Note that automated tests will be run on
    GitHub actions, but these must be initialized by a member of the team.


Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code, just
make a pull request. Pull requests should be made to the ``develop`` branch (subject to change).

For merging, you should:

1. Include an example for use
2. Add a note to `CHANGELOG.rst` about the changes
3. Update the author list in `AUTHORS.rst` if applicable
4. Ensure that all checks passed (current checks include GitHub Actions)

If you don't have all the necessary Python versions available locally or have
trouble building all the testing environments, you can rely on GitHub Actions
to run the tests for each change you add in the pull request. Because testing
here will delay tests by other developers, please ensure that the code passes
all tests on your local system first.

Project Style Guidelines
------------------------

The `nvector` project follows the 
[Napoleon NumPy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/#google-vs-numpy) with type-hinting. 
A good example is the following:

```python
def add_function(x1: int, x2: float) -> float:
    """Adds two numbers

    Parameters
    ----------
    x1 : int
        An integer value
    x2 : float
        A floating-point value
    
    Returns
    -------
    float
        The sum of the inputs.
        
    Examples
    --------
    >>> add_function(
    ...     1,
    ...     2.
    ... )
    3.
    """
    return x1 + x2 
```
In the case you cannot type-hint, try to be as descriptive in the docstrings as possible. Try to add docstring
examples using the `>>>` and `...` notation. 

Other choices include: 

* Block and inline comments should use proper English grammar and punctuation
  except with single sentences in a block, which may then omit the
  final period.

Further stylistic choices will be evaluated later.
