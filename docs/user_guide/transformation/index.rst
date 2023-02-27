.. -*- mode: rst -*-

Variance Stabilizing Transformations
====================================

Feature-engine's variable transformers transform numerical variables with various
mathematical transformations.

Variable transformations are commonly used to spread the values of the original variables
over a wider value range. See the following illustration:

.. figure::  ../../images/Variable_Transformation.png
   :align:   center


Article
-------

We added a lot of information about **variance stabilizing transformations** in this
`article <https://www.blog.trainindata.com/variance-stabilizing-transformations-in-machine-learning/>`_.

**Note**

Note however, that improving the value spread is not always possible and it depends
on the nature of the variable.

**Transformers**

.. toctree::
   :maxdepth: 1

   LogTransformer
   LogCpTransformer
   ReciprocalTransformer
   ArcsinTransformer
   PowerTransformer
   BoxCoxTransformer
   YeoJohnsonTransformer
