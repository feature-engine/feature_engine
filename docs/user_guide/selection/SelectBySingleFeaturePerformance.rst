SelectBySingleFeaturePerformance
================================

The SelectBySingleFeaturePerformance()selects features based on the performance of
machine learning models trained using individual features. In other words, selects
features based on their individual performance, returned by estimators trained on
only that particular feature.

.. code:: python

    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from feature_engine.selection import SelectBySingleFeaturePerformance

    # load dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.DataFrame(diabetes_y)

    # initialize feature selector
    sel = SelectBySingleFeaturePerformance(
            estimator=LinearRegression(), scoring="r2", cv=3, threshold=0.01)

    # fit transformer
    sel.fit(X, y)

    sel.features_to_drop_

.. code:: python

    [1]

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


