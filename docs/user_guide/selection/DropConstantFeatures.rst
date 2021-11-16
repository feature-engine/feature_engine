.. _drop_constant:

.. currentmodule:: feature_engine.selection

DropConstantFeatures
====================


The :class:`DropConstantFeatures()` drops constant and quasi-constant variables from a dataframe.
By default, it drops only constant variables. Constant variables have a single
value. Quasi-constant variables have a single value in most of its observations.

This transformer works with numerical and categorical variables, and it offers a pretty straightforward
way of reducing the feature space. Be mindful though, that depending on the context, quasi-constant
variables could be useful.

**Example**

Let's see how to use :class:`DropConstantFeatures()` in an example with the Titanic dataset. We
first load the data and separate it into train and test:

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

Now, we set up the :class:`DropConstantFeatures()` to remove features that show the same
value in more than 70% of the observations:

.. code:: python

    # set up the transformer
    transformer = DropConstantFeatures(tol=0.7, missing_values='ignore')


With `fit()` the transformer finds the variables to drop:

.. code:: python

    # fit the transformer
    transformer.fit(X_train)

The variables to drop are stored in the attribute `features_to_drop_`:

.. code:: python

    transformer.features_to_drop_

.. code:: python

    ['parch', 'cabin', 'embarked']


We see in the following code snippets that for the variables parch and embarked, more
than 70% of the observations displayed the same value:

.. code:: python

    X_train['embarked'].value_counts() / len(X_train)

.. code:: python

    S    0.711790
    C    0.197598
    Q    0.090611
    Name: embarked, dtype: float64


71% of the passengers embarked in S.

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

77% of the passengers had 0 parent or child. Because of this, these features were
deemed constant and removed.

With `transform()`, we can go ahead and drop the variables from the data:

.. code:: python

    # transform the data
    train_t = transformer.transform(X_train)

More details
^^^^^^^^^^^^

In this Kaggle kernel we use :class:`DropConstantFeatures()` together with other feature selection algorithms:

- `Kaggle kernel <https://www.kaggle.com/solegalli/feature-selection-with-feature-engine>`_
