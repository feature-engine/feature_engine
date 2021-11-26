.. -*- mode: rst -*-
.. currentmodule:: feature_engine.selection

Feature Selection
=================

Feature-engine's feature selection transformers are used to drop subsets of variables,
or in other words, to select subsets of variables. Feature-engine hosts selection
algorithms that are in general, not available in other libraries. These algorithms have
been gathered from data science competitions or used in the industry.

Feature-engine's transformers select features based on 2 strategies. They either select
features by looking at the features intrinsic characteristics like distributions or their
relationship with other features. Or they select features based on their impact on the
machine learning model performance.

In the following tables you find the algorithms that belong to either strategy.

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
:class:`DropHighPSIFeatures()`	                ×	                      √	            Drops features with high Population Stability Index
============================================ ======================= ============= ====================================================================================

Selection based on model performance
------------------------------------

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`SelectByShuffling()`	                ×	                      ×	            Selects features if shuffling their values causes a drop in model performance
:class:`SelectBySingleFeaturePerformance()`	    ×	                      ×	            Removes observations with missing data from the dataset
:class:`SelectByTargetMeanPerformance()`        √                         ×             Using the target mean as performance proxy, selects high performing features
:class:`RecursiveFeatureElimination()`          ×                         ×             Removes features recursively by evaluating model performance
:class:`RecursiveFeatureAddition()`             ×                         ×             Adds features recursively by evaluating model performance
============================================ ======================= ============= ====================================================================================


.. toctree::
   :maxdepth: 2
   :hidden:

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

Other Feature Selection Libraries
---------------------------------

For additional feature selection algorithms visit the following open-source libraries:

* `Scikit-learn selection <https://scikit-learn.org/stable/modules/feature_selection.html>`_
* `MLXtend selection <http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/>`_

Scikit-learn hosts multiple filter and embedded methods that select features based on
statistical tests or machine learning model derived importance. MLXtend hosts greedy
(wrapper) feature selection methods.
