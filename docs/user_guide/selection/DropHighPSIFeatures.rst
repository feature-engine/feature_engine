.. _psi_selection:

.. currentmodule:: feature_engine.selection

DropHighPSIFeatures
===================

Intro to :class:`DropHighPSIFeatures()`.

What does it do, when is it useful

**pls edit:**

DropHighPSIFeatures drops features which Population Stability Index (PSI) value is
above a given threshold. The PSI of a numerical feature is an indication of the
shift in its distribution; a feature with high PSI could therefore be considered
unstable.

In Credit Risk, eliminating features with high PSI is commonly done and usually
required by the Regulator.

When working with PSI, it is worth highlighting the following:

- The PSI is not symmetric; switching the order of the basis and test dataframes will lead to different PSI values.
- The number of bins has an impact on the PSI values.
- The PSI is a suitable metric for numerical features (i.e., either continuous or with high cardinality).
- For categorical or discrete features, the change in distributions is better assessed with Chi-squared.


Procedure
---------

Explain procedure with bullets


Example
-------

Case 1: split data based on proportions (split_frac)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain how it words

.. code:: python

    some python code

Next step of code

.. code:: python

    more code

Explanation of the output

Case 2: split data based on unique values (split_distinct)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain how it words. Give a meaninfuld example, eg, customer ids.

.. code:: python

    some python code

Next step of code

.. code:: python

    more code

Explanation of the output

Case 3: split data based on variable (cut_off is numerical value)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain how it words

.. code:: python

    some python code

Next step of code

.. code:: python

    more code

Explanation of the output


Case 4: split data based on variable (cut_off is list)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain how it words

.. code:: python

    some python code

Next step of code

.. code:: python

    more code

Explanation of the output

Case 5: split data based on variable (cut_off is date)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain how it words

.. code:: python

    some python code

Next step of code

.. code:: python

    more code

Explanation of the output

More details
^^^^^^^^^^^^

In this notebook, we show how to use :class:`DropHighPSIFeatures()`.

If we detail very well how to use all parameters here, we may not need a notebook.
Notebooks are located here:

https://github/feature-engine/feature-engine-examples/blob/main/selection/

