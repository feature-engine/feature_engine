.. -*- mode: rst -*-

Getting started with Feature-engine documentation
=================================================

Feature-engine documentation is built using `Sphinx <https://www.sphinx-doc.org>`_ and is hosted on `Read the Docs <https://readthedocs.org/>`_.

To learn more about Sphinx follow the `Sphinx Quickstart documentation <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.


Building the documentation
--------------------------

First, make sure you have properly installed Sphinx and the required dependencies.

1. If you haven't done so, in your virtual environment, from the root folder of the repository, install the requirements for the documentation::

        $ pip install -r docs/requirements.txt

2. To build the documentation (and test if it is working properly)::

    $ sphinx-build -b html docs build

This command tells sphinx that the documentation files are within the docs folder, and the html files should be placed in the
build folder.

If everything worked fine, you can navigate the html files located in build. Alternatively, you need to troubleshoot through
the error messages returned by sphinx.

Good luck and get in touch if stuck!