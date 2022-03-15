.. _single_feat_performance:

.. currentmodule:: feature_engine.selection

SelectBySingleFeaturePerformance
================================

The :class:`SelectBySingleFeaturePerformance()` selects features based on the performance of
machine learning models trained using individual features. That is, it selects
features based on their individual performance. In short, the selection algorithms works
as follows:

1. Train a machine learning model per feature (using only 1 feature)
2. Determine the performance metric of choice
3. Retain features which performance is above a threshold

If the parameter `threshold` is left to None, it will select features which performance is
above the mean performance of all features.

**Example**

Let's see how to use this transformer with the diabetes dataset that comes in Scikit-learn.
First, we load the data:

.. code:: python

    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from feature_engine.selection import SelectBySingleFeaturePerformance

    # load dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.DataFrame(diabetes_y)

Now, we start :class:`SelectBySingleFeaturePerformance()` to select features based on the
r2 returned by a Linear regression, using 3 fold cross-validation. We want to select features
which r2 > 0.01.

.. code:: python

    # initialize feature selector
    sel = SelectBySingleFeaturePerformance(
            estimator=LinearRegression(), scoring="r2", cv=3, threshold=0.01)

With `fit()` the transformer fits 1 model per feature, determines the performance and
selects the important features:

.. code:: python

    # fit transformer
    sel.fit(X, y)

The features that will be dropped are stored in an attribute:

.. code:: python

    sel.features_to_drop_

.. code:: python

    [1]

:class:`SelectBySingleFeaturePerformance()` also stores the performace of each one of the
models, in case we want to study those further:

..  code:: python

    sel.feature_performance_

.. code:: python

    {0: 0.029231969375784466,
     1: -0.003738551760264386,
     2: 0.336620809987693,
     3: 0.19219056680145055,
     4: 0.037115559827549806,
     5: 0.017854228256932614,
     6: 0.15153886177526896,
     7: 0.17721609966501747,
     8: 0.3149462084418813,
     9: 0.13876602125792703}

With `transform()` we go ahead and remove the features from the dataset:

.. code:: python

    # drop variables
    Xt = sel.transform(X)


More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Select-by-Single-Feature-Performance.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
