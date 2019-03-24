.. feature_engine documentation master file, created by
   sphinx-quickstart on Wed Jan 10 14:43:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to feature engine's documentation
==========================================

Feature Engine is a python library that contains several transformers to engineer features for use
in machine learning models.

The transformers work with scikit-learn like functionality. The first learn the imputing or encoding
methods from the training sets, and subsequently transform the dataset.

Currently the transformers include functionality for:

- Missing value imputation
- Categorical variable encoding
- Outlier removal
- Discretisation
- Numerical Variable Transformation



.. toctree::
   :maxdepth: 2
   :caption: Table of Contents
      
   MeanMedianImputer
   RandomSampleImputer
   EndTailImputer
   AddNaNBinaryImputer
   CategoricalVariableImputer
   FrequentCategoryImputer
   ArbitraryNumberImputer
   
   CountFrequencyCategoricalEncoder
   OrdinalCategoricalEncoder 
   MeanCategoricalEncoder
   WoERatioCategoricalEncoder
   OneHotCategoricalEncoder
   RareLabelCategoricalEncoder

   Windsorizer
   ArbitraryOutlierCapper
   
   EqualFrequencyDiscretiser
   EqualWidthDiscretiser
   DecisionTreeDiscretiser
   
   LogTransformer
   ReciprocalTransformer
   ExponentialTransformer
   BoxCoxTransformer




