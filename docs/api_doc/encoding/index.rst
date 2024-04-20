.. -*- mode: rst -*-

Categorical Encoding
====================

Feature-engine's categorical encoders replace the categories of the variable with
estimated or arbitrary numbers.

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

Feature-engine's categorical encoders encode only variables of type categorical or
object by default. From version 1.1.0, you have the option to set the parameter
`ignore_format` to True to make the transformers also accept numerical variables as
input.


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

Other categorical encoding libraries
------------------------------------

For additional categorical encoding transformations, visit the open-source package
`Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_.
