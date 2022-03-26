.. -*- mode: rst -*-
.. _selection_user_guide:

.. currentmodule:: feature_engine.selection


Feature Selection
=================

Feature-engine's feature selection transformers identify features with low predictive
performance and drop them from the dataset. To our knowledge, the feature selection
algorithms supported by Feature-engine are not yet available in other libraries. These
algorithms have been gathered from data science competitions or used in the industry.


Selection Mechanism Overview
----------------------------

Feature-engine's transformers select features based on 2 mechanism. The first mechanism
involves selecting features based on the features intrinsic characteristics like distributions
or their relationship with other features. The second mechanism involves selecting features
based on their impact on the machine learning model performance. In this context, features
are evaluated individually or as part of a feature group by different algorithms.

.. figure::  ../../images/selectionChart.png
   :align:   center

   Selection mechanisms - Overview

For example, in the first pillar, features will be selected based on the diversity of their
values, changes in their distribution or their relation to other features. This way,
features that show the same value in all or almost all the observations will be dropped,
features which distribution changes in time will be dropped, or duplicated or correlated
features will be dropped.

Algorithms that select features based on individual feature performance will select features
by either training a machine learning model using an individual feature, or estimating model
performance with a single feature using a prediction proxy.

Algorithms that select features based on their performance within a group of variables, will
normally train a model with all the features, and then remove or add or shuffle a feature and
re-evaluate the model performance.

These methods are normally geared to improve the overall performance of the final machine learning model
as well as reducing the feature space.


Selectors Characteristics Overview
----------------------------------

Some Feature-engine's selectors work with categorical variables off-the-shelf and/or allow
missing data in the variables. These gives you the opportunity to quickly screen features
before jumping into any feature engineering.

In the following tables we highlight the main Feature-engine selectors characteristics:

Selection based on feature characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`DropFeatures()`                         √	                      √	            Drops arbitrary features determined by user
:class:`DropConstantFeatures()`  	            √	                      √	            Drops constant and quasi-constant features
:class:`DropDuplicateFeatures()`                √	                      √             Drops features that are duplicated
:class:`DropCorrelatedFeatures()`               ×	                      √	            Drops features that are correlated
:class:`SmartCorrelatedSelection()`	            ×	                      √	            From a correlated feature group drops the less useful features
:class:`DropHighPSIFeatures()`	                ×	                      √	            Drops features with high Population Stability Index
============================================ ======================= ============= ====================================================================================

Selection based on model performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`SelectByShuffling()`	                ×	                      ×	            Selects features if shuffling their values causes a drop in model performance
:class:`SelectBySingleFeaturePerformance()`	    ×	                      ×	            Removes observations with missing data from the dataset
:class:`SelectByTargetMeanPerformance()`        √                         ×             Using the target mean as performance proxy, selects high performing features
:class:`RecursiveFeatureElimination()`          ×                         ×             Removes features recursively by evaluating model performance
:class:`RecursiveFeatureAddition()`             ×                         ×             Adds features recursively by evaluating model performance
============================================ ======================= ============= ====================================================================================

In short, selection procedures that require training a machine learning model from Scikit-learn
require numerical variables without missing data. Selection procedures based on correlation work
only with numerical variables but allow missing data. Methods that determine duplication or
the number of unique values can work with both numerical and categorical variables and support
missing data as well.

The :class:`SelectBySingleFeaturePerformance()` uses the target mean value as proxy for prediction,
replacing categories or variable intervals by these values and then determining a performance metric.
Thus, it is suitable for both categorical and numerical variables. In its current implementation,
it does not support missing data.

:class:`DropHighPSIFeatures()` allows to remove features with changes in their distribution. This is done by
splitting the input dataframe in two parts and comparing the distribution of each feature in the two
parts. The metric used to assess distribution shift is the Population Stability Index (PSI). Removing
unstable features may lead to more robust models. In fields like Credit Risk Modelling, the Regulator
often requires the PSI of the final feature set to be below are given threshold.

Throughout the user guide, you will find more details about each of the feature selection procedures.

Feature Selection Algorithms
----------------------------

Click below to find more details on how to use each one of the transformers.

.. toctree::
   :maxdepth: 1

   DropFeatures
   DropConstantFeatures
   DropDuplicateFeatures
   DropCorrelatedFeatures
   SmartCorrelatedSelection
   DropHighPSIFeatures
   SelectByShuffling
   SelectBySingleFeaturePerformance
   SelectByTargetMeanPerformance
   RecursiveFeatureElimination
   RecursiveFeatureAddition


Additional Resources
--------------------

More details about feature selection can be found in the following resources:

- `Feature Selection Online Course <https://courses.trainindata.com/p/feature-selection-for-machine-learning>`_
- `Feature Selection for Machine Learning: A comprehensive Overview <https://trainindata.medium.com/feature-selection-for-machine-learning-a-comprehensive-overview-bd571db5dd2d>`_
