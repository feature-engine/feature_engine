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
variables have missing data in the train set. If not, it ignores them during
`transform()`.

It is recommended to use `missing_only=True` when not passing a list of variables to
impute.

Below a code example using the House Prices Dataset (more details about the dataset
:ref:`here <datasets>`).

First, let's load the data and separate it into train and test:

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

Now, we set up the imputer to remove observations if they have missing data in any of
the variables indicated in the list.

.. code:: python

    # set up the imputer
    missingdata_imputer = DropMissingData(variables=['LotFrontage', 'MasVnrArea'])

    # fit the imputer
    missingdata_imputer.fit(X_train)


Now, we can go ahead and add the missing indicators:

.. code:: python

    # transform the data
    train_t= missingdata_imputer.transform(X_train)
    test_t= missingdata_imputer.transform(X_test)

We can explore the number of observations with NA in the variable `LotFrontage` before
the imputation:

.. code:: python

    # Number of NA before the transformation
    X_train['LotFrontage'].isna().sum()

.. code:: python

    189

And after the imputation we should not have observations with NA:

.. code:: python

    # Number of NA after the transformation:
    train_t['LotFrontage'].isna().sum()

.. code:: python

    0

We can go ahead and compare the shapes of the different dataframes, before and after
the imputation, and we will see that the imputed data has less observations, because
those with NA in any of the 2 variables of interest were removed.

.. code:: python

    # Number of rows before and after transformation
    print(X_train.shape)
    print(train_t.shape)

.. code:: python

    (1022, 79)
    (829, 79)

Drop partially complete rows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default behaviour of :class:`DropMissingData()` will drop rows in NA is present in
any of the variables indicated in the list.

We have the option of dropping rows only if a certain percentage of values is missing
across all variables.

For example, if we set the parameter `threshold=0.5`, a row will be dropped if data is
missing in 50% of the variables. If we set the parameter `threshold=0.01`, a row will
be dropped if data is missing in 1% of the variables. If we set the parameter
`threshold=1`, a row will be dropped if data is missing in all the variables.


More details
^^^^^^^^^^^^
In the following Jupyter notebook you will find more details on the functionality of the
:class:`DropMissingData()`, including how to select numerical variables automatically.

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/imputation/DropMissingData.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
