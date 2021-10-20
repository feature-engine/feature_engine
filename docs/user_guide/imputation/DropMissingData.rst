.. _drop_missing_data:

.. currentmodule:: feature_engine.imputation

DropMissingData
===============


The :class:`DropMissingData()` will delete rows containing missing values. It provides
similar functionality to `pandas.drop_na()`. The transformer has however some
advantages over pandas:

- it learns and stores the variables for which the rows with na should be deleted
- it can be used within the Scikit-learn pipeline

It works with numerical and categorical variables. You can pass a list of variables to
impute, or the transformer will select and impute all variables.

The trasformer has the option to learn the variables with missing data in the train set,
and then remove observations with NA only in those variables. Or alternatively remove
observations with NA in all variables. You can change the behaviour using the parameter
`missing_only`.

This means that if you pass a list of variables to impute and set `missing_only=True`,
and some of the variables in your list do not have missing data in the train set,
missing data will not be removed during transform for those particular variables. In
other words, when `missing_only=True`, the transformer "double checks" that the entered
variables have missing data in the train set. If not, it ignores it during
`transform()`.

It is recommended to use `missing_only=True` when not passing a list of variables to
impute.

Below a code example using the House Prices Dataset (more details about the dataset
:ref:`here <datasets>`).

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

More details
^^^^^^^^^^^^

Check also this `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/imputation/DropMissingData.ipynb>`_

