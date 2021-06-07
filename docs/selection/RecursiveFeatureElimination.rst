RecursiveFeatureElimination
============================

API Reference
-------------

.. autoclass:: feature_engine.selection.RecursiveFeatureElimination
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
     
..  code:: python

    # the features to remove
    tr.features_to_drop_

..  code:: python

    [0, 6, 7, 9]
    
..  code:: python

    print(Xt.head())

..  code:: python

              1         3         5         2         8         4
    0  0.050680  0.021872 -0.034821  0.061696  0.019908 -0.044223
    1 -0.044642 -0.026328 -0.019163 -0.051474 -0.068330 -0.008449
    2  0.050680 -0.005671 -0.034194  0.044451  0.002864 -0.045599
    3 -0.044642 -0.036656  0.024991 -0.011595  0.022692  0.012191
    4 -0.044642  0.021872  0.015596 -0.036385 -0.031991  0.003935
    
