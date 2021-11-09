.. _drop_features:

.. currentmodule:: feature_engine.selection

DropFeatures
=============

The :class:`DropFeatures()` drops a list of variables indicated by the user from the original
dataframe. The user can pass a single variable as a string or list of variables to be
dropped.

**When is this transformer useful?**

Sometimes, we create new variables combining other variables in the dataset, for
example, we obtain the variable `age` by subtracting `date_of_application` from
`date_of_birth`. After we obtained our new variable, we do not need the date
variables in the dataset any more. Thus, we can add DropFeatures() in the Pipeline
to have these removed.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine.selection import DropFeatures

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
                data.drop(['survived', 'name'], axis=1),
                data['survived'], test_size=0.3, random_state=0)

    # original columns
    X_train.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin',
           'embarked', 'boat', 'body', 'home.dest'],
          dtype='object')


.. code:: python

    # set up the transformer
    transformer = DropFeatures(
        features_to_drop=['sibsp', 'parch', 'ticket', 'fare', 'body', 'home.dest']
    )

    # fit the transformer
    transformer.fit(X_train)

    # transform the data
    train_t = transformer.transform(X_train)

    train_t.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'cabin', 'embarked' 'boat'],
          dtype='object')


More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Drop-Arbitrary-Features.ipynb>`_