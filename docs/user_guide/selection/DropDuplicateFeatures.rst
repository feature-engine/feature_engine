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
    from feature_engine.datasets import load_titanic
    from feature_engine.selection import DropDuplicateFeatures

    data = load_titanic(
        handle_missing=True,
        predictors_only=True,
    )

    # Lets duplicate some columns
    data = pd.concat([data, data[['sex', 'age', 'sibsp']]], axis=1)
    data.columns = ['pclass', 'survived', 'sex', 'age',
                    'sibsp', 'parch', 'fare','cabin', 'embarked',
                    'sex_dup', 'age_dup', 'sibsp_dup']

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['survived'], axis=1),
        data['survived'],
        test_size=0.3,
        random_state=0,
    )

    print(X_train.head())

Below we see the resulting data:

.. code:: python

          pclass     sex        age  sibsp  parch     fare    cabin embarked  \
    501        2  female  13.000000      0      1  19.5000  Missing        S
    588        2  female   4.000000      1      1  23.0000  Missing        S
    402        2  female  30.000000      1      0  13.8583  Missing        C
    1193       3    male  29.881135      0      0   7.7250  Missing        Q
    686        3  female  22.000000      0      0   7.7250  Missing        Q

         sex_dup    age_dup  sibsp_dup
    501   female  13.000000          0
    588   female   4.000000          1
    402   female  30.000000          1
    1193    male  29.881135          0
    686   female  22.000000          0

Now, we set up :class:`DropDuplicateFeatures()` to find the duplications:

.. code:: python

    transformer = DropDuplicateFeatures()

With `fit()` the transformer finds the duplicated features:

.. code:: python

    transformer.fit(X_train)

The features that are duplicated and will be removed are stored by the transformer:

..  code:: python

    transformer.features_to_drop_

.. code:: python

    {'age_dup', 'sex_dup', 'sibsp_dup'}

With `transform()` we remove the duplicated variables:

.. code:: python

    train_t = transformer.transform(X_train)
    test_t = transformer.transform(X_test)

If we examine the variable names of the transformed dataset, we see that the duplicated
features are not present:

.. code:: python

    train_t.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked'], dtype='object')

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

For more details about this and other feature selection methods check out these resources:

- `Feature selection for machine learning <https://www.trainindata.com/p/feature-selection-for-machine-learning>`_, online course.
- `Feature selection in machine learning <https://leanpub.com/feature-selection-in-machine-learning>`_, book.
