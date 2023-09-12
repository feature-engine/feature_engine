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
    from feature_engine.selection import RecursiveFeatureElimination

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
    tr = RecursiveFeatureElimination(estimator=linear_model, scoring="r2", cv=3)

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

    {0: -0.0032800993162502845,
     9: -0.00028194870232089997,
     6: -0.0006751427734088544,
     7: 0.00013890056776355575,
     1: 0.01195652626644067,
     3: 0.02863360798239445,
     5: 0.012639242239088355,
     2: 0.06630359039334816,
     8: 0.10937354113435072,
     4: 0.024318355833473526}

:class:`RecursiveFeatureAddition` also stores the features that will be dropped based
n the given threshold.

..  code:: python

    # the features to drop
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


