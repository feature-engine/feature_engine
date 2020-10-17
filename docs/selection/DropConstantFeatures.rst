DropConstantFeatures
====================

The DropConstantFeatures() drops constant and quasi-constant variables from a dataframe.
By default, DropConstantFeatures drops only constant variables. This transformer works
with both numerical and categorical variables.

.. code:: python

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.selection import DropConstantFeatures

    # Load dataset
    def load_titanic():
            data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
            data = data.replace('?', np.nan)
            data['cabin'] = data['cabin'].astype(str).str[0]
            data['pclass'] = data['pclass'].astype('O')
            data['embarked'].fillna('C', inplace=True)
            return data

    # load data as pandas dataframe
    data = load_titanic()

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                data.drop(['survived', 'name', 'ticket'], axis=1),
                data['survived'], test_size=0.3, random_state=0)

    # set up the transformer
    transformer = DropConstantFeatures(tol=0.7)

    # fit the transformer
    transformer.fit(X_train)

    # transform the data
    train_t = transformer.transform(X_train)

    transformer.constant_features_

.. code:: python

    ['parch', 'cabin', 'embarked']

.. code:: python

    X_train['embarked'].value_counts() / len(X_train)

.. code:: python

    S    0.711790
    C    0.197598
    Q    0.090611
    Name: embarked, dtype: float64

.. code:: python

    X_train['parch'].value_counts() / len(X_train)

.. code:: python

    0    0.771834
    1    0.125546
    2    0.086245
    3    0.005459
    4    0.004367
    5    0.003275
    6    0.002183
    9    0.001092
    Name: parch, dtype: float64


API Reference
-------------

.. autoclass:: feature_engine.selection.DropConstantFeatures
    :members: