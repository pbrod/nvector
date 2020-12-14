"""
Install nvector
===============

If you have pip installed and are online, then simply type:

    $ pip install nvector

to get the lastest stable version. Using pip also has the advantage that all
requirements are automatically installed.

You can download nvector and all dependencies to a folder "pkg", by the following:

   $ pip install --download=pkg nvector

To install the downloaded nvector, just type:

   $ pip install --no-index --find-links=pkg nvector



Verifying installation
======================
To verify that nvector can be seen by Python, type ``python`` from your shell.
Then at the Python prompt, try to import nvector:

.. parsed-literal::

    >>> import nvector as nv
    >>> print(nv.__version__)
    |release|


To test if the toolbox is working correctly paste the following in an interactive
python session::

   import nvector as nv
   nv.test('--doctest-modules')

or

   $ py.test --pyargs nvector --doctest-modules

at the command prompt.

"""