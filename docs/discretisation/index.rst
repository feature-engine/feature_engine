.. -*- mode: rst -*-

Variable Discretisation
=======================

Feature-engine's variable discretisation transformers transform continuous numerical
variables into discrete variables. The discrete variables will contain contiguous
intervals in the case of the equal frequency and equal width transformers. The
Decision Tree discretiser will return a discrete variable, in the sense that the
new feature takes a finite number of values.

.. toctree::
   :maxdepth: 2

   EqualFrequencyDiscretiser
   EqualWidthDiscretiser
   ArbitraryDiscretiser
   DecisionTreeDiscretiser
