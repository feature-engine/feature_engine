.. -*- mode: rst -*-
.. currentmodule:: feature_engine.selection

Feature Selection
=================

Feature-engine's feature selection transformers are used to drop subsets of variables.
Or in other words to select subsets of variables.

**Summary of Feature-engine's selectors main characteristics**

============================================ ======================= ============= ====================================================================================
    Transformer                                Categorical variables   Allows NA	    Description
============================================ ======================= ============= ====================================================================================
:class:`DropFeatures()`                         √	                      √	            Drops arbitrary features determined by user
:class:`DropConstantFeatures()`  	            √	                      √	            Drops constant and quasi-constant features
:class:`DropDuplicateFeatures()`                √	                      √             Drops features that are duplicated
:class:`DropCorrelatedFeatures()`               ×	                      √	            Drops features that are correlated
:class:`SmartCorrelatedSelection()`	            ×	                      √	            From a correlated feature group drops the less useful features
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
   SelectByShuffling
   SelectBySingleFeaturePerformance
   SelectByTargetMeanPerformance
   RecursiveFeatureElimination
   RecursiveFeatureAddition


Other Feature Selection Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For additional feature selection algorithms visit the following open-source libraries:

* `Scikit-learn selection <https://scikit-learn.org/stable/modules/feature_selection.html>`_
* `MLXtend selection <http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/>`_
