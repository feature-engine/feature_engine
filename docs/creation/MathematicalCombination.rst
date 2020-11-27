MathematicalCombination
=======================

API Reference
-------------

.. autoclass:: feature_engine.creation.MathematicalCombination
    :members:


Example
-------

MathematicalCombination() applies basic mathematical operations to multiple
features, returning one or more additional features as a result. Tha is, it sums,
multiplies, takes the average, maximum, minimum or standard deviation of a group
of variables and returns the result into new variables.

In this example, we sum 2 variables from the houseprices dataset.

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.creation import MathematicalCombination

    data = pd.read_csv('houseprice.csv').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Id', 'SalePrice'], axis=1),
        data['SalePrice'],
        test_size=0.3,
        random_state=0
    )

    math_combinator = MathematicalCombination(
        variables_to_combine=['LotFrontage', 'LotArea'],
        math_operations = ['sum'],
        new_variables_names = ['LotTotal']
    )

    math_combinator.fit(X_train, y_train)
    X_train_ = math_combinator.transform(X_train)

.. code:: python

    print(math_combinator.combination_dict_)

.. code:: python

    {'LotTotal': 'sum'}

.. code:: python

    print(X_train_.loc[:,['LotFrontage', 'LotArea', 'LotTotal']].head())

.. code:: python

          LotFrontage  LotArea  LotTotal
    64            0.0     9375    9375.0
    682           0.0     2887    2887.0
    960          50.0     7207    7257.0
    1384         60.0     9060    9120.0
    1100         60.0     8400    8460.0


