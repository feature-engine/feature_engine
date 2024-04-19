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
    y = pd.Series(diabetes_y)

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

If we now print the transformed data, we see that the features above were removed.

..  code:: python

    print(Xt.head())

..  code:: python

              0         2         3         4         5         6         7  \
    0  0.038076  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592
    1 -0.001882 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493
    2  0.085299  0.044451 -0.005670 -0.045599 -0.034194 -0.032356 -0.002592
    3 -0.089063 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309
    4  0.005383 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592

              8         9
    0  0.019907 -0.017646
    1 -0.068332 -0.092204
    2  0.002861 -0.025930
    3  0.022688 -0.009362
    4 -0.031988 -0.046641


Additional resources
--------------------

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Select-by-Single-Feature-Performance.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.

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