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

To set up `nvector` on your local host for development, you only need the Git application. You can download Git using
your distributions preferred method (dnf, yum, apt-get, brew) or using GitHub Desktop. 

The following are optional, but recommended.

* The CPython interpreter version 3.13
   * You can install it using official binaries, pyenv, or any Anaconda-like distribution 
* The `PDM <https://pdm-project.org/latest/>`_ application for locking, building, and testing.
* Dedicated IDE like PyCharm, Spyder, or `VSCodium <https://vscodium.com/>`_ to name a few. These allow more flexible git
  control.


Development Steps
-----------------

1. `Fork nvector on GitHub <https://github.com/pbrod/nvector>`_

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
   pdm use -i 3.13  # Automatically searches for Python3.13
   pdm use /path/to/python3.13  # Instead specify path to Python3.13
   pdm install -L pdm.lock 
   ```
    
    Now you can run tests

    ```shell
    pdm run pytest 
    ```
   
   If you have multiple Pythons installed to your path (3.9+), then you can use nox

   ```shell
   pdm run nox
   ```
   
5. For linting the source code you can use `ruff <https://pypi.org/project/ruff/#description>`_:

    ```shell
    ruff format ./src
    ```
    
6. Update/add documentation (in ``docs``), if relevant.
   
7. Add your name to the ``AUTHORS.rst`` file as an author.

8. Commit your changes. Using the command line would be:

   ```shell
   git add <FILE1> <FILE2> ... <FILEN>
   git commit -m "<type>(<scope>): <subject> <BLANK LINE> <body> <BLANK LINE> <footer>"
   ```

   See :ref:`commit-message-guidelines`.


9. Once you are happy with the local changes, push to GitHub:

   ```
   git push origin name-of-your-bugfix-or-feature
   ```
   
   Note that each push will trigger the Continuous Integration workflow. Check the ``Actions`` tab on your fork 
   repository home like github.com/<USER>/nvector/actions

10. Submit a pull request through the GitHub website. Pull requests should be
    made to the ``develop`` branch (subject to change).  Note that automated tests will be run on
    GitHub actions, but these must be initialized by a member of the team.

.. _commit-message-guidelines:

Commit message guidelines
-------------------------
The `nvector` project  uses python-semantic-release for automating the releases.
By analyzing the commit messages it takes care of incrementing the version number
and update the changelog as well as publish the package.

Therefore the commit messages must follow the 
`angular commit message style <https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commits>`_
This leads to more readable messages that are easy to follow when looking through the 
project history.
Each commit message consists of a ``header``, a ``body`` and a ``footer``. 
The header has a special format that includes a ``type``, a ``scope`` and a ``subject``::

    <type>(<scope>): <subject>
    <BLANK LINE>
    <body>
    <BLANK LINE>
    <footer>

The ``header`` is mandatory and the ``scope`` of the ``header`` is optional.

Any line of the commit message cannot be longer than 100 characters! 
This allows the message to be easier to read on GitHub as well as in various git tools.
The ``body`` or ``footer`` can begin with BREAKING CHANGE: 
followed by a short description to create a major release.

The ``type`` must be one of the following:

   * feat: A new feature
   * fix: A bug fix
   * docs: Documentation only changes
   * style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
   * refactor: A code change that neither fixes a bug nor adds a feature
   * perf: A code change that improves performance
   * test: Adding missing or correcting existing tests
   * chore: Changes to the build process or auxiliary tools and libraries such as documentation generation

The ``scope`` could be anything specifying place of the commit change. 
For example $core, $objects, $rotation, etc...

The ``subject`` contains succinct description of the change:

  * use the imperative, present tense: "change" not "changed" nor "changes"
  * don't capitalize first letter
  * no dot (.) at the end

The ``body`` should include the motivation for the change and contrast this 
with previous behavior. Use the imperative, present tense: 
"change" not "changed" nor "changes". 

The ``footer`` should contain any information about Breaking Changes and is 
also the place to reference GitHub issues that this commit closes.

Breaking Changes should start with the word BREAKING CHANGE: with a space or two newlines. 
The rest of the commit message is then used for this.


Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code, just
make a pull request. Pull requests should be made to the ``develop`` branch (subject to change).

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
