.. -*- mode: rst -*-
.. _encoding_user_guide:

.. currentmodule:: feature_engine.encoding


Categorical Encoding
====================

Feature-engine's categorical encoders replace variable strings by estimated or
arbitrary numbers. The following image summarizes the main encoder’s functionality.

**Summary of Feature-engine's encoders characteristics**

================================= ============ ================= ============== ===============================================================
    Transformer                    Regression	 Classification	   Multi-class    Description
================================= ============ ================= ============== ===============================================================
:class:`OneHotEncoder()`	           √	            √               √         Adds dummy variables to represent each category
:class:`OrdinalEncoder()`	           √	            √    	        √         Replaces categories with an integer
:class:`CountFreuencyEncoder()`	       √	            √               √         Replaces categories with their count or frequency
:class:`MeanEncoder()`                 √	            √               x         Replaces categories with the targe mean value
:class:`WoEEncoder()`	               x	            √	            x         Replaces categories with the weight of the evidence
:class:`DecisionTreeEncoder()`	       √	            √     	        √         Replaces categories with the predictions of a decision tree
:class:`RareLabelEncoder()`	           √	            √     	        √         Groups infrequent categories into a single one
================================= ============ ================= ============== ===============================================================


Feature-engine's categorical encoders work only with categorical variables by default.
From version 1.1.0, you have the option to set the parameter ignore_format to False,
and make the transformers also accept numerical variables as input.

**Monotonicity**

Most Feature-engine's encoders will return, or attempt to return monotonic relationships
between the encoded variable and the target. A monotonic relationship is one in which
the variable value increases as the values in the other variable increase, or decrease.
See the following illustration as examples:

.. figure::  ../../images/monotonic.png
   :align:   center
   :width: 400

Monotonic relationships tend to help improve the performance of linear models and build
shallower decision trees.

**Regression vs Classification**

Most Feature-engine's encoders are suitable for both regression and classification, with
the exception of the :class:`WoEEncoder()` and the :class:`PRatioEncoder()` which are
designed solely for **binary** classification.

**Multi-class classification**

Finally, some Feature-engine's encoders can handle multi-class targets off-the-shelf for
example the :class:`OneHotEncoder()`, the :class:CountFrequencyEncoder()` and the
:class:`DecisionTreeEncoder()`.

Note that while the :class:`MeanEncoder()` and the :class:`OrdinalEncoder()` will operate
with multi-class targets, but the mean of the classes may not be significant and this will
defeat the purpose of these encoding techniques.

**Encoders**

.. toctree::
   :maxdepth: 1

   OneHotEncoder
   CountFrequencyEncoder
   OrdinalEncoder
   MeanEncoder
   WoEEncoder
   DecisionTreeEncoder
   RareLabelEncoder
   StringSimilarityEncoder
