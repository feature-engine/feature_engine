.. -*- mode: rst -*-

Missing Data Imputation
=======================

Feature-engine's missing data imputers replace missing data by parameters estimated
from data or arbitrary values pre-defined by the user. The following image summarizes
the main imputer's functionality.

.. figure::  ../../images/summary/imputersSummary.png
   :align:   center

|

In this guide, you will find code snippets to quickly be able to apply the imputers
to your datasets, as well as general knowledge and guidance on the imputation
techniques.


Imputers
~~~~~~~~

.. toctree::
   :maxdepth: 1

   MeanMedianImputer
   ArbitraryNumberImputer
   EndTailImputer
   CategoricalImputer
   RandomSampleImputer
   AddMissingIndicator
   DropMissingData