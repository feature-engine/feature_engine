.. -*- mode: rst -*-
.. currentmodule:: feature_engine.discretisation

Discretisation
==============

Feature-engine's discretisation transformers transform continuous variables into
discrete features. This is accomplished, in general, by sorting the variable values
into continuous intervals.

**Summary**

=====================================  ========================================================================
      Transformer                           Functionality
=====================================  ========================================================================
:class:`EqualFrequencyDiscretiser()`     Sorts values into intervals with similar number of observations.
:class:`EqualWidthDiscretiser()`         Sorts values into intervals of equal size.
:class:`ArbitraryDiscretiser()`          Sorts values into intervals predefined by the user.
:class:`DecisionTreeDiscretiser()`       Replaces values by predictions of a decision tree, which are discrete.
:class:`GeometricWidthDiscretiser()`     Sorts variable into geometrical intervals.
=====================================  ========================================================================


.. toctree::
   :maxdepth: 1
   :hidden:

   EqualFrequencyDiscretiser
   EqualWidthDiscretiser
   ArbitraryDiscretiser
   DecisionTreeDiscretiser
   GeometricWidthDiscretiser

Additional transformers for discretisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For discretisation using K-means, check Scikit-learn's
`KBinsDiscretizer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html>`_.
