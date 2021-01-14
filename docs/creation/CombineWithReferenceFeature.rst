CombineWithReferenceFeature
=======================

API Reference
-------------

.. autoclass:: feature_engine.creation.CombineWithReferenceFeature
    :members:


Example
-------

CombineWithReferenceFeature() applies basic binary operations to multiple
features, returning one or more additional features as a result. That is, it substract or
divide of a group of variables and returns the result into new variables.

In this example, we subtract 2 variables from the house prices dataset.

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.creation import CombineWithReferenceFeature

    data = pd.read_csv('houseprice.csv').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0
    )

    combinator = CombineWithReferenceFeature(
        variables_to_combine=['LotArea'],
        reference_variables=['LotFrontage'],
        operations = ['sub'],
        new_variables_names = ['LotTotal']
        )

    combinator.fit(X_train, y_train)
    X_train_ = combinator.transform(X_train)

    print(X_train_[["LotTotal","LotFrontage","LotArea"]].head())

.. code:: python

        LotTotal  LotFrontage  LotArea
    64      9375.0          0.0     9375
    682     2887.0          0.0     2887
    960     7157.0         50.0     7207
    1384    9000.0         60.0     9060
    1100    8340.0         60.0     8400
