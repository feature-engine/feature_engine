DropDuplicateFeatures
=====================

API Reference
-------------

.. autoclass:: feature_engine.selection.DropDuplicateFeatures
    :members:


Example
-------

The DropDuplicateFeatures() finds and removes duplicated variables from a dataframe.
The user can pass a list of variables to examine, or alternatively the selector will
examine all variables in the data set.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine.selection import DropDuplicateFeatures

    def load_titanic():
            data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
            data = data.replace('?', np.nan)
            data['cabin'] = data['cabin'].astype(str).str[0]
            data = data[['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'cabin', 'embarked']]
            data = pd.concat([data, data[['sex', 'age', 'sibsp']]], axis=1)
            data.columns = ['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'cabin', 'embarked',
                            'sex_dup', 'age_dup', 'sibsp_dup']
            return data

    # load data as pandas dataframe
    data = load_titanic()
    data.head()

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                data.drop(['survived'], axis=1),
                data['survived'], test_size=0.3, random_state=0)

    # set up the transformer
    transformer = DropDuplicateFeatures()

    # fit the transformer
    transformer.fit(X_train)

    # transform the data
    train_t = transformer.transform(X_train)

    train_t.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'cabin', 'embarked'], dtype='object')

..  code:: python

    transformer.duplicated_features_

.. code:: python

    {'age_dup', 'sex_dup', 'sibsp_dup'}

.. code:: python

    transformer.duplicated_feature_sets_

.. code:: python

    [{'sex', 'sex_dup'}, {'age', 'age_dup'}, {'sibsp', 'sibsp_dup'}]


