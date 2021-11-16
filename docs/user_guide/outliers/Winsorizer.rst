.. _winsorizer:

.. currentmodule:: feature_engine.outliers

Winsorizer
==========

The :class:`Winsorizer()` caps maximum and/or minimum values of a variable at automatically
determined values. The minimum and maximum values can be calculated in 1 of 3 different ways:

Gaussian limits:

- right tail: mean + 3* std
- left tail: mean - 3* std

IQR limits:

- right tail: 75th quantile + 3* IQR
- left tail:  25th quantile - 3* IQR

where IQR is the inter-quartile range: 75th quantile - 25th quantile.

percentiles or quantiles:

- right tail: 95th percentile
- left tail:  5th percentile

**Example**

Let's cap some outliers in the Titanic Dataset. First, let's load the data and separate
it into train and test:

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine.outliers import Winsorizer

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

Now, we will set the :class:`Winsorizer()` to cap outliers at the right side of the
distribution only (param `tail`). We want the maximum values to be determined using the
mean value of the variable (param `capping_method`) plus 3 times the standard deviation
(param `fold`). And we only want to cap outliers in 2 variables, which we indicate in a
list.

.. code:: python

    # set up the capper
    capper = Winsorizer(capping_method='gaussian', tail='right', fold=3, variables=['age', 'fare'])

    # fit the capper
    capper.fit(X_train)

With `fit()`, the :class:`Winsorizer()` finds the values at which it should cap the variables.
These values are stored in its attribute:

.. code:: python

    capper.right_tail_caps_

.. code:: python

	{'age': 67.49048447470315, 'fare': 174.78162171790441}

We can now go ahead and censor the outliers:

.. code:: python

    # transform the data
    train_t= capper.transform(X_train)
    test_t= capper.transform(X_test)
    
If we evaluate now the maximum of the variables in the transformed datasets, they should
coincide with the values observed in the attribute `right_tail_caps_`:

.. code:: python

    train_t[['fare', 'age']].max()

.. code:: python

    fare    174.781622
    age      67.490484
    dtype: float64

More details
^^^^^^^^^^^^

You can find more details about the :class:`Winsorizer()` functionality in the following
notebook:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/outliers/Winsorizer.ipynb>`_
