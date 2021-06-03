DropMissingData
===============

API Reference
-------------

.. autoclass:: feature_engine.imputation.DropMissingData
    :members:

Example
-------

DropMissingData() deletes rows with missing values. It works with numerical and
categorical variables. You can pass a list of variables to impute, or the transformer
will select and impute all variables. The trasformer has the option to learn the
variables with missing data in the train set, and then remove observations with NA only
in those variables.

.. code:: python

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.imputation import DropMissingData

    # Load dataset
    data = pd.read_csv('houseprice.csv')

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0)

    # set up the imputer
    missingdata_imputer = DropMissingData(variables=['LotFrontage', 'MasVnrArea'])

    # fit the imputer
    missingdata_imputer.fit(X_train)

    # transform the data
    train_t= missingdata_imputer.transform(X_train)
    test_t= missingdata_imputer.transform(X_test)

    # Number of NA before the transformation
    X_train['LotFrontage'].isna().sum()

.. code:: python

    189

.. code:: python

    # Number of NA after the transformation:
	train_t['LotFrontage'].isna().sum()

.. code:: python

    0

.. code:: python

    # Number of rows before and after transformation
    print(X_train.shape)
    print(train_t.shape)

.. code:: python

    (1022, 79)
    (829, 79)



