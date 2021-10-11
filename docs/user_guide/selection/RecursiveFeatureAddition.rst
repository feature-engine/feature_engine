RecursiveFeatureAddition
========================

API Reference
-------------

.. autoclass:: feature_engine.selection.RecursiveFeatureAddition
    :members:


Example
-------

.. code:: python

    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from feature_engine.selection import RecursiveFeatureElimination

    # load dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.DataFrame(diabetes_y)

    # initialize linear regresion estimator
    linear_model = LinearRegression()

    # initialize feature selector
    tr = RecursiveFeatureElimination(estimator=linear_model, scoring="r2", cv=3)

    # fit transformer
    Xt = tr.fit_transform(X, y)

    # get the initial linear model performance, using all features
    tr.initial_model_performance_

.. code:: python

    0.488702767247119

..  code:: python

    # Get the performance drift of each feature
    tr.performance_drifts_

..  code:: python

    {4: 0,
     8: 0.2837159006046677,
     2: 0.1377700238871593,
     5: 0.0023329006089969906,
     3: 0.0187608758643259,
     1: 0.0027994385024313617,
     7: 0.0026951300105543807,
     6: 0.002683967832484757,
     9: 0.0003040126429713075,
     0: -0.007386876030245182}

..  code:: python

    # the features to drop
    tr.features_to_drop_

..  code:: python

    [0, 6, 7, 9]

..  code:: python

    print(Xt.head())

..  code:: python

              4         8         2         3
    0 -0.044223  0.019908  0.061696  0.021872
    1 -0.008449 -0.068330 -0.051474 -0.026328
    2 -0.045599  0.002864  0.044451 -0.005671
    3  0.012191  0.022692 -0.011595 -0.036656
    4  0.003935 -0.031991 -0.036385  0.021872

