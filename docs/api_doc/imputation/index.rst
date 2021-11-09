.. -*- mode: rst -*-

.. currentmodule:: feature_engine.imputation

Missing Data Imputation
=======================

Feature-engine's missing data imputers replace missing data by parameters estimated
from data or arbitrary values pre-defined by the user.


**Summary of Feature-engine's imputers main characteristics**

================================== ===================== ======================= ====================================================================================
    Transformer                     Numerical variables	  Categorical variables	    Description
================================== ===================== ======================= ====================================================================================
:class:`MeanMedianImputer()`	        √	                 ×	                    Replaces missing values by the mean or median
:class:`ArbitraryNumberImputer()`	    √	                 x	                    Replaces missing values by an arbitrary value
:class:`EndTailImputer()`	            √	                 ×	                    Replaces missing values by a value at the end of the distribution
:class:`CategoricalImputer()`           √	                 √	                    Replaces missing values by the most frequent category or by an arbitrary value
:class:`RandomSampleImputer()`	        √	                 √	                    Replaces missing values by random value extractions from the variable
:class:`AddMissingIndicator()`	        √	                 √	                    Adds a binary variable to flag missing observations
:class:`DropMissingData()`	            √	                 √	                    Removes observations with missing data from the dataset
================================== ===================== ======================= ====================================================================================


The :class:`CategoricalImputer()` performs procedures suitable for categorical variables. From
version 1.1.0 it also accepts numerical variables as input, for those cases were
categorical variables by nature are coded as numeric.


.. toctree::
   :maxdepth: 2
   :hidden:

   MeanMedianImputer
   ArbitraryNumberImputer
   EndTailImputer
   CategoricalImputer
   RandomSampleImputer
   AddMissingIndicator
   DropMissingData