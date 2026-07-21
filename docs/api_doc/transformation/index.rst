.. -*- mode: rst -*-

Variance Stabilising Transformations
====================================

Feature-engine's variable transformers transform numerical variables with various
mathematical transformations.

.. toctree::
   :maxdepth: 1

   LogTransformer
   LogCpTransformer
   ReciprocalTransformer
   ArcsinTransformer
   ArcSinhTransformer
   PowerTransformer
   BoxCoxTransformer
   YeoJohnsonTransformer


Transformers in other Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These and additional transformations can be obtained with the following scikit-learn
classes:

* `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_
* `PowerTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html>`_

Note that scikit-learn classes return NumPy arrays and are applied to the entire dataset.
