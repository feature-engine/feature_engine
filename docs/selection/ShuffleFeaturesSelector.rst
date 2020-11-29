SelectByShuffling
=================

The SelectByShuffling() selects important features if permutation their values
at random produces a decrease in the initial model performance. See API below for
more details into its functionality.

.. code:: python

    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from feature_engine.selection import ShuffleFeaturesSelector

    # load dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.DataFrame(diabetes_y)

    # initialize linear regresion estimator
    linear_model = LinearRegression()

    # initialize feature selector
    tr = SelectByShuffling(estimator=linear_model, scoring="r2", cv=3)

    # fit transformer
    Xt = tr.fit_transform(X, y)

    tr.initial_model_performance_


.. code:: python

    0.488702767247119

..  code:: python

    tr.performance_drifts_

.. code:: python

    {0: -0.02368121940502793,
     1: 0.017909161264480666,
     2: 0.18565460365508413,
     3: 0.07655405817715671,
     4: 0.4327180164470878,
     5: 0.16394693824418372,
     6: -0.012876023845921625,
     7: 0.01048781540981647,
     8: 0.3921465005640224,
     9: -0.01427065640301245}

.. code:: python

    tr.selected_features_

.. code:: python

    [1, 2, 3, 4, 5, 7, 8]

.. code:: python

    print(Xt.head())

.. code:: python

              1         2         3         4         5         7         8
    0  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.002592  0.019908
    1 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163 -0.039493 -0.068330
    2  0.050680  0.044451 -0.005671 -0.045599 -0.034194 -0.002592  0.002864
    3 -0.044642 -0.011595 -0.036656  0.012191  0.024991  0.034309  0.022692
    4 -0.044642 -0.036385  0.021872  0.003935  0.015596 -0.002592 -0.031991
    None

API Reference
-------------

.. autoclass:: feature_engine.selection.SelectByShuffling
    :members: