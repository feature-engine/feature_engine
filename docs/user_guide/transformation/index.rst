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

================================== ========================= =====================================================
    Transformer                       Limitations     	  	    Description
================================== ========================= =====================================================
:class:`LogTransformer()`	         Not valid for x<=0         Applies natural or decimal logarithm.
:class:`LogCpTransformer()`	               None	                Applies logarithm after adding a constant value.
:class:`ReciprocalTransformer()`     Not defined for x=0        Applies the reciprocal transformation: 1/x.
:class:`ArcsinTransformer()`             None                   Applies the inverse sine function.
:class:`ArcSinhTransformer()`	     	 None                   Applies the inverse hyperbolic sine function.
:class:`PowerTransformer()`	        	 None                   Applies any power transformation x = x**n.
:class:`BoxCoxTransformer()`	     Not defined for x<=0       Applies the Box-Cox transformation.
:class:`YeoJohnsonTransformer()`         None                   Applies the Yeo-Johnson transformation.
================================== ========================= =====================================================

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
