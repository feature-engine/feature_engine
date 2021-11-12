.. _drop_features:

.. currentmodule:: feature_engine.selection

DropFeatures
=============

The :class:`DropFeatures()` drops a list of variables indicated by the user from the original
dataframe. The user can pass a single variable as a string or list of variables to be
dropped.

:class:`DropFeatures()` offers similar functionality to `pandas.dataframe.drop <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>`_,
but the difference is that :class:`DropFeatures()` can be integrated into a Scikit-learn
pipeline.


**When is this transformer useful?**

Sometimes, we create new variables combining other variables in the dataset, for
example, we obtain the variable `age` by subtracting `date_of_application` from
`date_of_birth`. After we obtained our new variable, we do not need the date
variables in the dataset any more. Thus, we can add :class:`DropFeatures()` in the Pipeline
to have these removed.

**Example**

Let's see how to use :class:`DropFeatures()` in an example with the Titanic dataset. We
first load the data and separate it into train and test:

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

Now, we go ahead and print the dataset column names:

.. code:: python

    # original columns
    X_train.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin',
           'embarked', 'boat', 'body', 'home.dest'],
          dtype='object')

Now, with :class:`DropFeatures()` we can very easily drop a group of variables. Below
we set up the transformer to drop a list of 6 variables:

.. code:: python

    # set up the transformer
    transformer = DropFeatures(
        features_to_drop=['sibsp', 'parch', 'ticket', 'fare', 'body', 'home.dest']
    )

    # fit the transformer
    transformer.fit(X_train)

With `fit()` this transformer does not learn any parameter. We can go ahead and remove
the variables as follows:

.. code:: python

    # transform the data
    train_t = transformer.transform(X_train)

And now, if we print the variable names of the transformed dataset, we see that it has
been reduced:

.. code:: python

    train_t.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'cabin', 'embarked' 'boat'],
          dtype='object')


More details
^^^^^^^^^^^^

In this Kaggle kernel we feature 3 different end-to-end machine learning pipelines using
:class:`DropFeatures()`:

- `Kaggle Kernel <https://www.kaggle.com/solegalli/feature-engineering-and-model-stacking>`_
