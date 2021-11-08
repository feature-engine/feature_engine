.. -*- mode: rst -*-

Missing Data Imputation
=======================

Feature-engine's missing data imputers replace missing data by parameters estimated
from data or arbitrary values pre-defined by the user.

.. figure::  ../../images/summary/imputersSummary.png
   :align:   center

   Summary of Feature-engine's imputers main characteristics

The `CategoricalImputer` performs procedures suitable for categorical variables. From
version 1.1.0 it also accepts numerical variables as input, for those cases were
categorical variables by nature are coded as numeric.


.. toctree::
   :maxdepth: 2

   MeanMedianImputer
   ArbitraryNumberImputer
   EndTailImputer
   CategoricalImputer
   RandomSampleImputer
   AddMissingIndicator
   DropMissingData