.. -*- mode: rst -*-
.. currentmodule:: feature_engine.selection

Feature Selection
=================

Feature-engine's feature selection transformers are used to drop subsets of variables
with low predictive value. Feature-engine hosts selection algorithms that are, in general,
not available in other libraries. These algorithms have been gathered from data science
competitions or used in the industry.

Feature-engine's transformers select features based on different strategies. Some algorithms
remove constant or quasi-constant features. Some algorithms remove duplicated or correlated
variables. Some algorithms select features based on a machine learning model performance.
Some transformers implement selection procedures used in finance. And some transformers support
functionality that has been developed in the industry or in data science competitions.

In the following tables you find the algorithms that belong to each category.

Selection based on feature characteristics
------------------------------------------

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`DropFeatures()`                         √	                      √	            Drops arbitrary features determined by user
:class:`DropConstantFeatures()`  	            √	                      √	            Drops constant and quasi-constant features
:class:`DropDuplicateFeatures()`                √	                      √             Drops features that are duplicated
:class:`DropCorrelatedFeatures()`               ×	                      √	            Drops features that are correlated
:class:`SmartCorrelatedSelection()`	            ×	                      √	            From a correlated feature group drops the less useful features
:class:`MRMR()`                  	            √	                      ×	            Selects features based on the MRMR framework
============================================ ======================= ============= ====================================================================================

Selection based on a machine learning model
-------------------------------------------

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`SelectBySingleFeaturePerformance()`	    ×	                      ×	            Selects features based on single feature model performance
:class:`RecursiveFeatureElimination()`          ×                         ×             Removes features recursively by evaluating model performance
:class:`RecursiveFeatureAddition()`             ×                         ×             Adds features recursively by evaluating model performance
============================================ ======================= ============= ====================================================================================

Selection methods commonly used in finance
------------------------------------------

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`DropHighPSIFeatures()`	                ×	                      √	            Drops features with high Population Stability Index
:class:`SelectByInformationValue()`	            √                         x             Drops features with low information value
============================================ ======================= ============= ====================================================================================

Alternative feature selection methods
-------------------------------------

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`SelectByShuffling()`	                ×	                      ×	            Selects features if shuffling their values causes a drop in model performance
:class:`SelectByTargetMeanPerformance()`        √                         ×             Using the target mean as performance proxy, selects high performing features
:class:`ProbeFeatureSelection()`                ×                         ×             Selects features who importance is greater than those of random variables
============================================ ======================= ============= ====================================================================================


.. toctree::
   :maxdepth: 1
   :hidden:

   DropFeatures
   DropConstantFeatures
   DropDuplicateFeatures
   DropCorrelatedFeatures
   SmartCorrelatedSelection
   SelectBySingleFeaturePerformance
   RecursiveFeatureElimination
   RecursiveFeatureAddition
   DropHighPSIFeatures
   SelectByInformationValue
   SelectByShuffling
   SelectByTargetMeanPerformance
   ProbeFeatureSelection
   MRMR

Other Feature Selection Libraries
---------------------------------

For additional feature selection algorithms visit the following open-source libraries:

* `Scikit-learn selection <https://scikit-learn.org/stable/modules/feature_selection.html>`_
* `MLXtend selection <http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/>`_

Scikit-learn hosts multiple filter and embedded methods that select features based on
statistical tests or machine learning model derived importance. MLXtend hosts greedy
(wrapper) feature selection methods.
