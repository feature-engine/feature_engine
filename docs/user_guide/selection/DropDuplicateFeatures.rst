.. _drop_duplicate:

.. currentmodule:: feature_engine.selection

DropDuplicateFeatures
=====================

The :class:`DropDuplicateFeatures()` finds and removes duplicated variables from a dataframe.
Duplicated features are identical features, regardless of the variable or column name. If
they show the same values for every observation, then they are considered duplicated.

The transformer will automatically evaluate all variables, or alternatively, you can pass a
list with the variables you wish to have examined. And it works with numerical and categorical
features.

**Example**

Let's see how to use :class:`DropDuplicateFeatures()` in an example with the Titanic dataset.
These dataset does not have duplicated features, so we will add a few manually:

.. code:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from feature_engine.selection import DropDuplicateFeatures
    from feature_engine.datasets import load_titanic

    data = load_titanic(handle_missing=True)

    data['cabin'] = data['cabin'].astype(str).str[0]
    data = data[['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'cabin', 'embarked']]
    data = pd.concat([data, data[['sex', 'age', 'sibsp']]], axis=1)
    data.columns = ['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'cabin', 'embarked',
                    'sex_dup', 'age_dup', 'sibsp_dup']

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                data.drop(['survived'], axis=1),
                data['survived'], test_size=0.3, random_state=0)

Now, we set up :class:`DropDuplicateFeatures()` to find the duplications:

.. code:: python

    # set up the transformer
    transformer = DropDuplicateFeatures()

With `fit()` the transformer finds the duplicated features, With `transform()` it removes
them:

.. code:: python

    # fit the transformer
    transformer.fit(X_train)

    # transform the data
    train_t = transformer.transform(X_train)

If we examine the variable names of the transformed dataset, we see that the duplicated
features are not present:

.. code:: python

    train_t.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'cabin', 'embarked'], dtype='object')

The features that are removed are stored in the transformer's attribute:

..  code:: python

    transformer.features_to_drop_

.. code:: python

    {'age_dup', 'sex_dup', 'sibsp_dup'}

And the transformer also stores the groups of duplicated features, which could be useful
if we have groups where more than 2 features are identical.

.. code:: python

    transformer.duplicated_feature_sets_

.. code:: python

    [{'sex', 'sex_dup'}, {'age', 'age_dup'}, {'sibsp', 'sibsp_dup'}]


More details
^^^^^^^^^^^^

In this Kaggle kernel we use :class:`DropDuplicateFeatures()` together with other feature selection algorithms:

- `Kaggle kernel <https://www.kaggle.com/solegalli/feature-selection-with-feature-engine>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
