.. _recursive_elimination:

.. currentmodule:: feature_engine.selection

RecursiveFeatureElimination
============================

:class:`RecursiveFeatureElimination` implements recursive feature elimination. Recursive
feature elimination (RFE) is a backward feature selection process. In Feature-engine's
implementation of RFE, a feature will be kept or removed based on the performance of a
machine learning model without that feature. This differs from Scikit-learn's implementation of
`RFE <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>`_
where a feature will be kept or removed based on the feature importance.

This technique begins by building a model on the entire set of variables, then calculates and
stores a model performance metric, and finally computes an importance score for each variable.
Features are ranked by the model’s `coef_` or `feature_importances_` attributes.

In the next step, the least important feature is removed, the model is re-built, and a new performance
metric is determined. If this performance metric is worse than the original one, then,
the feature is kept, (because eliminating the feature clearly caused a drop in model
performance) otherwise, it removed.

The procedure removes now the second to least important feature, trains a new model, determines a
new performance metric, and so on, until it evaluates all the features, from the least
to the most important.

Note that, in Feature-engine's implementation of RFE, the feature importance is used
just to rank features and thus determine the order
in which the features will be eliminated. But whether to retain a feature is determined
based on the decrease in the performance of the model after the feature elimination.

By recursively eliminating features, RFE attempts to eliminate dependencies and
collinearity that may exist in the model.

**Parameters**

Feature-engine's RFE has 2 parameters that need to be determined somewhat arbitrarily by
the user: the first one is the machine learning model which performance will be evaluated. The
second is the threshold in the performance drop that needs to occur, to remove a feature.

RFE is not machine learning model agnostic, this means that the feature selection depends on
the model, and different models may have different subsets of optimal features. Thus, it is
recommended that you use the machine learning model that you finally intend to build.

Regarding the threshold, this parameter needs a bit of hand tuning. Higher thresholds will
of course return fewer features.

**Example**

Let's see how to use this transformer with the diabetes dataset that comes in Scikit-learn.
First, we load the data:


.. code:: python

    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from feature_engine.selection import RecursiveFeatureElimination

    # load dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.Series(diabetes_y)

Now, we set up :class:`RecursiveFeatureElimination` to select features based on the r2
returned by a Linear Regression model, using 3 fold cross-validation. In this case,
we leave the parameter `threshold` to the default value which is 0.01.

.. code:: python

    # initialize linear regresion estimator
    linear_model = LinearRegression()

    # initialize feature selector
    tr = RecursiveFeatureElimination(estimator=linear_model, scoring="r2", cv=3)

With `fit()` the model finds the most useful features, that is, features that when removed
cause a drop in model performance bigger than 0.01. With `transform()`, the transformer
removes the features from the dataset.

.. code:: python

    # fit transformer
    Xt = tr.fit_transform(X, y)


:class:`RecursiveFeatureElimination` stores the performance of the model trained using all
the features in its attribute:

.. code:: python

    # get the initial linear model performance, using all features
    tr.initial_model_performance_
    
.. code:: python

    0.488702767247119

:class:`RecursiveFeatureElimination`  also stores the change in the performance caused by
removing every feature.

..  code:: python

    # Get the performance drift of each feature
    tr.performance_drifts_
    
..  code:: python

    {0: -0.0032796652347705235,
     9: -0.00028200591588534163,
     6: -0.0006752869546966522,
     7: 0.00013883578730117252,
     1: 0.011956170569096924,
     3: 0.028634492035512438,
     5: 0.012639090879036363,
     2: 0.06630127204137715,
     8: 0.1093736570697495,
     4: 0.024318093565432353}

:class:`RecursiveFeatureElimination` also stores the features that will be dropped based
n the given threshold.

..  code:: python

    # the features to remove
    tr.features_to_drop_

..  code:: python

    [0, 6, 7, 9]

If we now print the transformed data, we see that the features above were removed.

..  code:: python

    print(Xt.head())

..  code:: python

              1         2         3         4         5         8
    0  0.050680  0.061696  0.021872 -0.044223 -0.034821  0.019907
    1 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163 -0.068332
    2  0.050680  0.044451 -0.005670 -0.045599 -0.034194  0.002861
    3 -0.044642 -0.011595 -0.036656  0.012191  0.024991  0.022688
    4 -0.044642 -0.036385  0.021872  0.003935  0.015596 -0.031988


Additional resources
--------------------

More details on recursive feature elimination in this article:

- `Recursive feature elimination with Python <https://www.blog.trainindata.com/recursive-feature-elimination-with-python/>`_

For more details about this and other feature selection methods check out these resources:

For more details about this and other feature selection methods check out these resources:


.. figure::  ../../images/fsml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-selection-for-machine-learning

   Feature Selection for Machine Learning

|
|
|
|
|
|
|
|
|
|

Or read our book:

.. figure::  ../../images/fsmlbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://leanpub.com/feature-selection-in-machine-learning

   Feature Selection in Machine Learning

|
|
|
|
|
|
|
|
|
|
|
|
|
|

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.