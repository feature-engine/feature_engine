ShuffleFeaturesSelector
=======================

The ShuffleFeaturesSelector() selects features by determining the drop in machine
learning model performance when each feature's values are randomly shuffled.
The user can pass a list of variables to examine, or alternatively the selector will
examine all variables in the data set.

.. code:: python

    def load_diabetes_data():

        diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
        data = pd.DataFrame(diabetes_X)
        target = pd.DataFrame(diabetes_y)

        return data, target

    data, target = load_diabetes_data()

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=0)

    # initialize linear regresion estimator
    linear_model = LinearRegression()
    # initialize ShuffleFeaturesSelector 
    transformer = ShuffleFeaturesSelector(
        estimator=model,
        scoring='r2',
        cv = 3)

    # Fit the selector 
    X = transformer.fit_transform(X_train, y_train)
    
    # Get the initial model performance
    transformer.initial_model_performance_
    
.. code:: python

    0.7536374800253398
    
..  code:: python

    # Get the model performance drifts
    transformer.performance_drifts_

.. code:: python

    {0: -0.02357321349215452,
     1: 0.02338737974865923,
     2: 0.19373678730180866,
     3: 0.07535196370036401,
     4: 0.4661882256235008,
     5: 0.1511853641781783,
     6: -0.01230682744138506,
     7: 0.006374262665166719,
     8: 0.4566400050483824,
     9: -0.015247265065878424}
     
..  code:: python

    # Get the transformer variables
    transformer.variables

.. code:: python

    ['index', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
     
..  code:: python

    # Get the transformer selected features
    transformer.selected_features_

.. code:: python

    ['index', 2, 3, 4, 5, 8]
    
API Reference
-------------

.. autoclass:: feature_engine.selection.ShuffleFeaturesSelector
    :members: