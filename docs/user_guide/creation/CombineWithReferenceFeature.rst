CombineWithReferenceFeature
===========================

CombineWithReferenceFeature() combines a group of variables with a group of reference
variables utilizing basic mathematical operations (subtraction, division, addition and
multiplication), returning one or more additional features in the dataframe as a result.

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
        new_variables_names = ['LotPartial']
        )

    combinator.fit(X_train, y_train)
    X_train_ = combinator.transform(X_train)

    print(X_train_[["LotPartial","LotFrontage","LotArea"]].head())

.. code:: python

        LotTotal  LotFrontage  LotArea
    64      9375.0          0.0     9375
    682     2887.0          0.0     2887
    960     7157.0         50.0     7207
    1384    9000.0         60.0     9060
    1100    8340.0         60.0     8400
