DropFeatures
=============
The DropFeatures() drops a list of variables from the original dataframe. The user can pass a single variable as
a string or list of variables to be dropped.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine.feature_selection import DropFeatures

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
    transformer = DropFeatures(
        features_to_drop=['survived', 'name', 'sibsp', 'parch',
                          'ticket', 'fare', 'body', 'home.dest']
    )

    # fit the transformer
    transformer.fit(X_train)

    # transform the data
    train_t = transformer.transform(X_train)

    train_t.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'cabin', 'embarked' 'boat'],
          dtype='object')


API Reference
-------------

.. autoclass:: feature_engine.selection.DropFeatures
    :members: