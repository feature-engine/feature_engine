.. _recursive_addition:

.. currentmodule:: feature_engine.selection

RecursiveFeatureAddition
========================

:class:`RecursiveFeatureAddition` implements recursive feature addition. Recursive
feature addition (RFA) is a forward feature selection process.

This technique begins by building a model on the entire set of variables and computing
an importance score for each variable. Features are ranked by the model’s `coef_` or
`feature_importances_` attributes.

In the next step, it trains a model only using the feature with the highest importance and
stores the model performance.

Then, it adds the second most important, trains a new model and determines a new performance
metric. If the performance increases beyond the threshold, compared to the previous model,
then that feature is important and will be kept. Otherwise, that feature is removed.

It proceeds to evaluate the next most important feature, and so on, until all features
are evaluated.

Note that feature importance is used just to rank features and thus determine the order
in which the features will be added. But whether to retain a feature is determined based
on the increase in the performance of the model after the feature addition.

**Parameters**

Feature-engine's RFA has 2 parameters that need to be determined somewhat arbitrarily by
the user: the first one is the machine learning model which performance will be evaluated. The
second is the threshold in the performance increase that needs to occur, to keep a feature.

RFA is not machine learning model agnostic, this means that the feature selection depends on
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
    from feature_engine.selection import RecursiveFeatureAddition

    # load dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.Series(diabetes_y)

Now, we set up :class:`RecursiveFeatureAddition` to select features based on the r2
returned by a Linear Regression model, using 3 fold cross-validation. In this case,
we leave the parameter `threshold` to the default value which is 0.01.

.. code:: python

    # initialize linear regresion estimator
    linear_model = LinearRegression()

    # initialize feature selector
    tr = RecursiveFeatureAddition(estimator=linear_model, scoring="r2", cv=3)

With `fit()` the model finds the most useful features, that is, features that when added
cause an increase in model performance bigger than 0.01. With `transform()`, the transformer
removes the features from the dataset.

.. code:: python

    # fit transformer
    Xt = tr.fit_transform(X, y)

:class:`RecursiveFeatureAddition` stores the performance of the model trained using all
the features in its attribute:

.. code:: python

    # get the initial linear model performance, using all features
    tr.initial_model_performance_

.. code:: python

    0.488702767247119

:class:`RecursiveFeatureAddition`  also stores the change in the performance caused by
adding each feature.

..  code:: python

    # Get the performance drift of each feature
    tr.performance_drifts_

..  code:: python

    {4: 0,
     8: 0.28371458794131676,
     2: 0.1377714799388745,
     5: 0.0023327265047610735,
     3: 0.018759914615172735,
     1: 0.0027996354657459643,
     7: 0.002695149440021638,
     6: 0.002683934134630306,
     9: 0.000304067408860742,
     0: -0.007387230783454768}


:class:`RecursiveFeatureAddition` also stores the features that will be dropped based
n the given threshold.

..  code:: python

    # the features to drop
    tr.features_to_drop_

..  code:: python

    [0, 1, 5, 6, 7, 9]

If we now print the transformed data, we see that the features above were removed.

..  code:: python

    print(Xt.head())

..  code:: python

              2         3         4         8
    0  0.061696  0.021872 -0.044223  0.019907
    1 -0.051474 -0.026328 -0.008449 -0.068332
    2  0.044451 -0.005670 -0.045599  0.002861
    3 -0.011595 -0.036656  0.012191  0.022688
    4 -0.036385  0.021872  0.003935 -0.031988


Additional resources
--------------------

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