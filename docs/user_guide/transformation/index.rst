.. -*- mode: rst -*-

Variance Stabilising Transformations
====================================

Feature-engine's variance stabilising transformers transform numerical variables with various
mathematical operations, like logarithm, power, reciprocal, and so on.

Variable transformations are commonly used to spread the values of the original variables
over a wider value range and help meet the assumptions of several statistical models.
See the following illustration:

.. figure::  ../../images/Variable_Transformation.png
   :align:   center


.. tip::

    To learn more about `**variance stabilising transformations** <https://www.blog.trainindata.com/variance-stabilizing-transformations-in-machine-learning/>`_
    and their role in statistics and  machine learning, check out our very detailed
    `article <https://www.blog.trainindata.com/variance-stabilizing-transformations-in-machine-learning/>`_ in our
    Train in Data Blog.

Supported transformations
-------------------------

================================ ================================================ ============================================================================================= ====================
Transformer                      Description                                      Suitable for                                                                                  Limitations
================================ ================================================ ============================================================================================= ====================
:class:`LogTransformer()`        Applies natural or decimal logarithm.            Positive continuous variables with right skew.                                                Not valid for x<=0
:class:`LogCpTransformer()`      Applies logarithm after adding a constant value. Continuous variables with a right skew.                                                       None
:class:`ReciprocalTransformer()` Applies the reciprocal transformation: 1/x.      Variables representing ratios or proportions, like tons per acre.                             Not defined for x=0
:class:`ArcsinTransformer()`     Applies the arcsin square root transformation.   Probabilities or proportion variables with values between 0 and 1.                            0<= x <= 1
:class:`ArcSinhTransformer()`    Applies the inverse hyperbolic sine function.    Similar to log but retaining zero values in a variable.                                       None
:class:`PowerTransformer()`      Applies any power transformation x = x**n.       Square root is suitable for count variables. Other powers vary.                               None
:class:`BoxCoxTransformer()`     Applies the Box-Cox transformation.              Positive continuous variables when the optimal transformation is unknown.                     Not defined for x<=0
:class:`YeoJohnsonTransformer()` Applies the Yeo-Johnson transformation.          Continuous variables with zero or negative values when the optimal transformation is unknown. None
================================ ================================================ ============================================================================================= ====================

.. note::

    Improving the value spread is not always possible and it depends on the nature of
    the variable.

Transformers
------------

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
