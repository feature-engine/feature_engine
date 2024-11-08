.. _drop_duplicate:

.. currentmodule:: feature_engine.selection

DropDuplicateFeatures
=====================

Duplicate features are columns in a dataset that are identical, or, in other words, that
contain exactly the same values. Duplicate features can be introduced accidentally, either
through poor data management processes or during data manipulation.

For example, duplicated new records can be created by one-hot encoding a categorical
variable or by adding missing data indicators. We can also accidentally generate duplicate
records when we merge different data sources that show some variable overlap.

Checking for and removing duplicate features is a standard procedure in any data analysis
workflow that helps us reduce the dimension of the dataset quickly and ensure data quality.
In Python, we can find duplicate values in an attribute table very easily with Pandas.
Dropping those duplicate features, however, requires a few more lines of code.

Feature-engine aims to accelerate the process of data validation by finding and removing
duplicate features with the :class:`DropDuplicateFeatures()` class, which is part of the
selection API.

:class:`DropDuplicateFeatures()` does exactly that; it finds and removes duplicated variables
from a dataframe. DropDuplicateFeatures() will automatically evaluate all variables, or
alternatively, you can pass a list with the variables you wish to have examined. And it
works with numerical and categorical features alike.

So let’s see how to set up :class:`DropDuplicateFeatures()`.

**Example**

In this demo, we will use the Titanic dataset and introduce a few duplicated features
manually:

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


We then split the data into a training and a testing set:

.. code:: python

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

As expected, the variables `sex` and `sex_dup` have duplicate field values throughout all
the rows. The same is true for the variables `age` and `age_dup`.

Now, we set up :class:`DropDuplicateFeatures()` to find the duplicate features:

.. code:: python

    transformer = DropDuplicateFeatures()

With `fit()` the transformer finds the duplicated features:

.. code:: python

    transformer.fit(X_train)

The features that are duplicated and will be removed are stored in the `features_to_drop_`
attribute:

..  code:: python

    transformer.features_to_drop_

.. code:: python

    {'age_dup', 'sex_dup', 'sibsp_dup'}

With `transform()` we remove the duplicated variables:

.. code:: python

    train_t = transformer.transform(X_train)
    test_t = transformer.transform(X_test)

We can go ahead and check the variables in the transformed dataset, and we will see that
the duplicated features are not there any more:

.. code:: python

    train_t.columns

.. code:: python

    Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked'], dtype='object')

The transformer also stores the groups of duplicated features, which is useful for data
analysis and validation.

.. code:: python

    transformer.duplicated_feature_sets_

.. code:: python

    [{'sex', 'sex_dup'}, {'age', 'age_dup'}, {'sibsp', 'sibsp_dup'}]


Additional resources
--------------------

In this Kaggle kernel we use :class:`DropDuplicateFeatures()` in a pipeline with other
feature selection algorithms:

- `Kaggle kernel <https://www.kaggle.com/solegalli/feature-selection-with-feature-engine>`_

For more details about this and other feature selection methods check out these resources:


.. figure::  ../../images/fsml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-selection-for-machine-learning

   Feature Selection for Machine Learning

|
|
|
|
|
|
|
|
|
|

Or read our book:

.. figure::  ../../images/fsmlbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-selection-in-machine-learning-book

   Feature Selection in Machine Learning

|
|
|
|
|
|
|
|
|
|
|
|
|
|

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.