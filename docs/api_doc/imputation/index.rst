.. -*- mode: rst -*-


Missing Data Imputation
=======================

Feature-engine's missing data imputers replace missing data by parameters estimated
from data or arbitrary values pre-defined by the user.

The following table summarizes each imputer's functionality:

================================== ===================== ======================= ====================================================================================
    Transformer                     Numerical variables	  Categorical variables	    Description
================================== ===================== ======================= ====================================================================================
:class:`MeanMedianImputer()`	        √	                 ×	                    Replaces missing values with the mean or median
:class:`ArbitraryNumberImputer()`	    √	                 x	                    Replaces missing values with an arbitrary value
:class:`EndTailImputer()`	            √	                 ×	                    Replaces missing values with a value at the end of the distribution
:class:`CategoricalImputer()`           ×	                 √	                    Replaces missing values with the most frequent category or an arbitrary string
:class:`RandomSampleImputer()`	        √	                 √	                    Replaces missing values with random value extractions from the variable
:class:`AddMissingIndicator()`	        √	                 √	                    Adds a binary variable to flag missing observations
:class:`DropMissingData()`	            √	                 √	                    Removes observations with missing data from the dataset
================================== ===================== ======================= ====================================================================================

Imputers
--------

.. toctree::
   :maxdepth: 1

   MeanMedianImputer
   ArbitraryNumberImputer
   EndTailImputer
   CategoricalImputer
   RandomSampleImputer
   AddMissingIndicator
   DropMissingData