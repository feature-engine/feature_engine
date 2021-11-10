.. _target_mean_selection:

.. currentmodule:: feature_engine.selection


SelectByTargetMeanPerformance
=============================
    SelectByTargetMeanPerformance() uses the mean value of the target per category, or
    interval if the variable is numerical, as proxy for target estimation. With this
    proxy and the real target, the selector determines a performance metric for each
    feature, and then selects them based on this performance metric.

    SelectByTargetMeanPerformance() works with numerical and categorical variables.
    First, it eparates the training set into train and test sets. Then it works as
    follows:

    For each categorical variable:

    1. Determines the mean target value per category using the train set
    (equivalent to target mean encoding).

    2. Replaces the categories in the test set by the target mean values.

    3. Using the encoded variables and the real target calculates the roc-auc or r2.

    4. Selects the features which roc-auc or r2 is bigger than the indicated
    threshold.

    For each numerical variable:

    1- Discretises the variable into intervals of equal width or equal frequency
    (uses the discretisers of Feature-engine).

    2- Determines the mean value of the target per interval using the train set.

    3- Replaces the intervals in the test set, by the target mean values.

    4- Using the transformed variable and the real target calculates the roc-auc or r2.

    5- Selects the features which roc-auc or r2 is bigger than the indicated
    threshold.

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from feature_engine.selection import SelectByTargetMeanPerformance

    # load data
    data = pd.read_csv('../titanic.csv')

    # extract cabin letter
    data['cabin'] = data['cabin'].str[0]

    # replace infrequent cabins by N
    data['cabin'] = np.where(data['cabin'].isin(['T', 'G']), 'N', data['cabin'])

    # cap maximum values
    data['parch'] = np.where(data['parch']>3,3,data['parch'])
    data['sibsp'] = np.where(data['sibsp']>3,3,data['sibsp'])

    # cast variables as object to treat as categorical
    data[['pclass','sibsp','parch']] = data[['pclass','sibsp','parch']].astype('O')

    # separate train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['survived'], axis=1),
        data['survived'],
        test_size=0.3,
        random_state=0)


    # feature engine automates the selection for both categorical and numerical
    # variables
    sel = SelectByTargetMeanPerformance(
        variables=None,
        scoring="roc_auc_score",
        threshold=0.6,
        bins=3,
        strategy="equal_frequency", 
        cv=2,# cross validation
        random_state=1, #seed for reproducibility
    )

    # find important features
    sel.fit(X_train, y_train)

    sel.variables_categorical_

.. code:: python

    ['pclass', 'sex', 'sibsp', 'parch', 'cabin', 'embarked']

.. code:: python

    sel.variables_numerical_

.. code:: python

    ['age', 'fare']

.. code:: python

    sel.feature_performance_

.. code:: python

    {'pclass': 0.6802934787230475,
     'sex': 0.7491365252482871,
     'age': 0.5345141148737766,
     'sibsp': 0.5720480307315783,
     'parch': 0.5243557188989476,
     'fare': 0.6600883312700917,
     'cabin': 0.6379782658154696,
     'embarked': 0.5672382248783936}

.. code:: python

    sel.features_to_drop_

.. code:: python

    ['age', 'sibsp', 'parch', 'embarked']

.. code:: python

    # remove features
    X_train = sel.transform(X_train)
    X_test = sel.transform(X_test)

    X_train.shape, X_test.shape

.. code:: python

    ((914, 4), (392, 4))


More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Select-by-Target-Mean-Encoding.ipynb>`_
