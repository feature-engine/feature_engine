.. -*- mode: rst -*-

.. _contribute_docs:

Contribute Docs
===============

If you contribute a new transformer, or enhance the functionality of a current transformer,
most likely, you would have to add or update the documentation as well.

This is Feature-engine's documentation ecosystem:

- Feature-engine documentation is built using `Sphinx <https://www.sphinx-doc.org>`_ and is hosted on `Read the Docs <https://readthedocs.org/>`_.
- We use the `pydata sphinx theme <https://pypi.org/project/pydata-sphinx-theme/>`_.
- We follow `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_ for doscstring conventions and use `numpydoc docstring style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
- All documentation files are located within the `docs folder <https://github.com/feature-engine/feature_engine/tree/main/docs>`_ in the repository.

To learn more about Sphinx check the `Sphinx Quickstart documentation <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.

Documents organisation
----------------------

Feature-engine has just adopted Scikit-learn's documentation style, were we offer API
documentation, as well as, a User Guide with examples on how to use the different transformers.

The API documentation is built directly from the docstrings from each transformer. If you
are adding a new transformer, you need to reference it in a new rst file placed within
the `api_doc folder <https://github.com/feature-engine/feature_engine/tree/main/docs/api_doc>`_.

If you would like to add additional examples, you need to update the rst files located
in the `user_guide folder <https://github.com/feature-engine/feature_engine/tree/main/docs/user_guide>`_.

Docstrings
----------

The quickest way to get started with writing the transformer docstrings, is too look at the docstrings
of some of the classes we already have in Feature-engine. Then simply copy and paste
those docstrings and edit the bits that you need. If you copy and paste, make sure to delete
irrelevant parameters and methods.

Link a new transformer
----------------------

If you coded a new transformer from scratch, you need to update the following files to make
sure users can find information on how to use the class correctly:

Add the name of your transformer in these files:

- `Readme <https://github.com/feature-engine/feature_engine/blob/main/README.md>`_.
- `main index <https://github.com/feature-engine/feature_engine/blob/main/docs/index.rst>`_.
- `api index <https://github.com/feature-engine/feature_engine/tree/main/docs/api_doc/index.rst>`_.
- `user guide index <https://github.com/feature-engine/feature_engine/tree/main/docs/user_guide/index.rst>`_.

Add an rst file with the name of your transformer in these folders:

- `api_doc folder <https://github.com/feature-engine/feature_engine/tree/main/docs/api_doc>`_.
- `user_guide folder <https://github.com/feature-engine/feature_engine/tree/main/docs/user_guide>`_.

That's it!

Expand the User Guide
---------------------

You can add more examples or more details to our current User Guide examples. First, find
the relevant rst file for the transformer you would like to work with. Feel free to add more
details on the description of the method, expand the code showcasing other parameters or
whatever you see fit.

We normally run the code on jupyter notebooks, and then copy and paste the code and the
output in the rst files.

Build the documentation
-----------------------

To build the documentation, make sure you have properly installed Sphinx and the required
dependencies. If you set up the development environment as we described in the
:ref:`contribute code guide <contribute_code>`, you should have those installed already.

Alternatively, first activate your environment. Then navigate to the root folder of
Feature-engine. And now install the requirements for the documentation::

        $ pip install -r docs/requirements.txt

To build the documentation (and test if it is working properly) run::

    $ sphinx-build -b html docs build

This command tells sphinx that the documentation files are within the "docs" folder, and
the html files should be placed in the "build" folder.

If everything worked fine, you can open the html files located in build using your browser.
Alternatively, you need to troubleshoot through the error messages returned by sphinx.

Good luck and get in touch if stuck!