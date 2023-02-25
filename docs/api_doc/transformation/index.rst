.. -*- mode: rst -*-

Variance Stabilizing Transformations
====================================

Feature-engine's variable transformers transform numerical variables with various
mathematical transformations.

.. toctree::
   :maxdepth: 1

   LogTransformer
   LogCpTransformer
   ReciprocalTransformer
   ArcsinTransformer
   PowerTransformer
   BoxCoxTransformer
   YeoJohnsonTransformer


Transformers in other Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These and additional transformations can be obtained with the following Scikit-learn
classes:

* `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_
* `PowerTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html>`_

Note that Scikit-klearn classes return Numpy arrays and are applied to the entire dataset.
