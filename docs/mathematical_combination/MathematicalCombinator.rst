MathematicalCombinator
======================
    The MathematicalCombinator() applies basic mathematical operations across features,
    returning one or more additional features as a result.

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine import mathematical_combination as mc

    data = pd.read_csv('houseprice.csv').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['Id', 'SalePrice'], axis=1),
        data['SalePrice'],
        test_size=0.3,
        random_state=0
    )

    math_combinator = mc.MathematicalCombinator(
        variables=['LotFrontage', 'LotArea'],
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

API Reference
-------------

.. autoclass:: feature_engine.creation.MathematicalCombination
    :members:
