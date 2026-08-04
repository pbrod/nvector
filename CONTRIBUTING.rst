Contributing
============

Bug reports, feature suggestions, and other contributions are greatly appreciated!

Short version
-------------

* Submit bug reports and feature requests at 
  `GitHub <https://github.com/pbrod/nvector/issues>`_

* Make pull requests to the ``develop`` branch.

Bug reports
-----------

When `reporting a bug <https://github.com/pbrod/nvector/issues>`_ please
include:

* Your operating system name and version

* Any details about your local setup that might be helpful in troubleshooting

* Detailed steps to reproduce the bug


Feature requests and feedback
-----------------------------

The best way to send feedback is to file an issue at
`GitHub. <https://github.com/pbrod/nvector/issues>`_

If you are proposing a feature:

* Explain in detail how it would work.

* Keep the scope as narrow as possible, to make it easier to implement.

* Remember that this is a volunteer-driven project, and that code contributions
  are welcome :)


Development Prerequisites
-------------------------

To set up `nvector` on your local host for development, you only need Git.
You can install Git using your distribution's preferred method (dnf, yum,
apt-get, brew, etc.) or by using GitHub Desktop.

The following tools are recommended:

* Python 3.13 for development (Python 3.10 or newer is required).
* The `PDM <https://pdm-project.org/latest/>`_ package manager for dependency
  management, testing, and packaging.
* An IDE such as PyCharm, Spyder, VS Codium, or Visual Studio Code.

Development Steps
-----------------

1. Fork `nvector` on GitHub:

   https://github.com/pbrod/nvector

2. Clone your fork locally:

   .. code-block:: shell

       git clone git@github.com:<USER>/nvector.git

3. Create a development branch:

   .. code-block:: shell

       git checkout develop
       git checkout -b name-of-your-bugfix-or-feature

4. Install the development environment:

   .. code-block:: shell

       pdm use -i 3.13
       pdm install -d

   This installs the project together with the development dependencies.

   To run the full local validation suite:

   .. code-block:: shell

       pdm all-tests

   To run only the tests:

   .. code-block:: shell

       pdm run pytest

   If you have multiple Python versions installed, you can also run:

   .. code-block:: shell

       pdm run nox

5. Format and lint the source code:

   .. code-block:: shell

       pdm format
       pdm check-style
       pdm check-types

6. Update documentation in ``docs`` if relevant.

7. Consider adding your name to ``AUTHORS.rst`` for significant contributions.

8. Commit your changes:

   .. code-block:: shell

       git add <FILE1> <FILE2> ...
       git commit -m "<type>(<scope>): <subject>"

   See :ref:`commit-message-guidelines`.

9. Push your branch:

   .. code-block:: shell

       git push origin name-of-your-bugfix-or-feature

   Each push automatically triggers the GitHub Actions CI workflow.

10. Submit a pull request against the ``develop`` branch.


Release Tooling
---------------

Release tooling is installed separately from the normal development
dependencies.

To install release tools:

.. code-block:: shell

    pdm install -G release

This installs utilities such as:

* git-cliff
* pdm-bump

These tools are only required when preparing a release.

See ``RELEASE.md`` for the complete release workflow and publishing process.


.. _commit-message-guidelines:

Commit Message Guidelines
-------------------------

The `nvector` project uses Conventional Commits together with
`git-cliff` and `pdm-bump` for changelog generation and release
management.

The ``type`` should be one of:

* feat: A new feature
* fix: A bug fix
* docs: Documentation changes
* style: Formatting changes only
* refactor: Refactoring without feature or bug fixes
* perf: Performance improvements
* test: Test changes
* ci: Continuous integration changes
* chore: Build or tooling changes

Examples:

.. code-block:: text

    feat(core): add geodesic helper

.. code-block:: text

    fix(objects): correct distance calculation

.. code-block:: text

    ci: update GitHub Actions workflow


The ``scope`` identifies the area of the project affected by the change.

Examples include ``core``, ``objects``, ``rotation``, ``docs``, ``ci``, and
similar project components.

The ``subject`` contains a concise description of the change:

* Use the imperative, present tense: "change" not "changed" or "changes".
* Do not capitalize the first letter.
* Do not end the subject with a period.

The ``body`` should explain the motivation for the change and contrast it with
previous behavior. Use the imperative, present tense throughout.

The ``footer`` should contain information about breaking changes and references
to GitHub issues that are closed by the commit.

Breaking changes should begin with:

.. code-block:: text

    BREAKING CHANGE: description

The remainder of the footer should explain the impact of the change and any
required migration steps.


Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code, just
make a pull request. Pull requests should be made aginst the ``develop`` branch.

For merging, you should:

1. Include an example for use
2. Update the author list in `AUTHORS.rst` if applicable
3. Ensure that all checks passed (current checks include GitHub Actions)

If you don't have all the necessary Python versions available locally or have
trouble building all the testing environments, you can rely on GitHub Actions
to run the tests for each change you add in the pull request. Because testing
here will delay tests by other developers, please ensure that the code passes
all tests on your local system first.

Project Style Guidelines
------------------------

The `nvector` project follows the 
`Napoleon NumPy style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/#google-vs-numpy>`_ 
with type-hinting. 
A good example is the following:

.. code-block:: python

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

In the case you cannot type-hint, try to be as descriptive in the docstrings as possible. Try to add docstring
examples using the `>>>` and `...` notation. 

Other choices include: 

* Block and inline comments should use proper English grammar and punctuation
  except with single sentences in a block, which may then omit the
  final period.

Further stylistic choices will be evaluated later.
