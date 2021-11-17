.. -*- mode: rst -*-
.. currentmodule:: feature_engine.selection

Feature Selection
=================

Feature-engine's feature selection transformers are used to drop subsets of variables,
or in other words, to select subsets of variables. Feature-engine hosts selection
algorithms that are in general, not available in other libraries. These algorithms have
been gathered from data science competitions or used in the industry.

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

Feature-engine also hosts selection methods based on variable distributions. Currently,
selection based on Population Stability Index is available through the :class:`DropHighPSIFeatures()`.
Note that these methods may not necessarily enhance model performance, but may be necessary
to abide by regulations.

Selection Transformers
----------------------

.. toctree::
   :maxdepth: 2

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
   DropHighPSIFeatures


Other Feature Selection Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For additional feature selection algorithms visit the following open-source libraries:

* `Scikit-learn selection <https://scikit-learn.org/stable/modules/feature_selection.html>`_
* `MLXtend selection <http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/>`_

Scikit-learn hosts multiple filter and embedded methods, that select features based on
statistical tests or machine learning model derived importance. MLXtend hosts greedy
(wrapper) feature selection methods.
