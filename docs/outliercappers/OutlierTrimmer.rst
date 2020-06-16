OutlierTrimmer
==============

Removes values beyond predefined minimum and maximum values from the data set.
The minimum and maximum values can be calculated in 1 of 3 different ways:

Gaussian limits:
    right tail: mean + 3* std

    left tail: mean - 3* std

IQR limits:
    right tail: 75th quantile + 3* IQR

    left tail:  25th quantile - 3* IQR

where IQR is the inter-quartile range: 75th quantile - 25th quantile.

percentiles or quantiles:
    right tail: 95th percentile

    left tail:  5th percentile

See the API Reference for more details.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine import outlier_removers as outr

    # Load dataset
    def load_titanic():
        data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
        data = data.replace('?', np.nan)
        data['cabin'] = data['cabin'].astype(str).str[0]
        data['pclass'] = data['pclass'].astype('O')
        data['embarked'].fillna('C', inplace=True)
        data['fare'] = data['fare'].astype('float')
        data['fare'].fillna(data['fare'].median(), inplace=True)
        data['age'] = data['age'].astype('float')
        data['age'].fillna(data['age'].median(), inplace=True)
        return data

    data = load_titanic()

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
		data.drop(['survived', 'name', 'ticket'], axis=1),
		data['survived'], test_size=0.3, random_state=0)

    # set up the capper
    capper = outr.OutlierTrimmer(
        distribution='skewed', tail='right', fold=1.5, variables=['age', 'fare'])

    # fit the capper
    capper.fit(X_train)

    # transform the data
    train_t= capper.transform(X_train)
    test_t= capper.transform(X_test)

    capper.right_tail_caps_


.. code:: python

	{'age': 53.0, 'fare': 66.34379999999999}

.. code:: python

    train_t[['fare', 'age']].max()

.. code:: python

    fare    65.0
    age     53.0
    dtype: float64


API Reference
-------------

.. autoclass:: feature_engine.outlier_removers.OutlierTrimmer
    :members:
