.. _target_mean_selection:

.. currentmodule:: feature_engine.selection


SelectByTargetMeanPerformance
=============================

:class:`SelectByTargetMeanPerformance()` uses the mean value of the target per unique
category, or per interval if the variable is numerical, as proxy for prediction. And with
this prediction, it determines a performance metric against the target.

:class:`SelectByTargetMeanPerformance()` splits the data into two halves. The first half
is used as training set to determine the mappings from category to mean target value or from
interval to mean target value. The second half is the test set, where the categories and
intervals will be mapped to the determined values, these will be considered a prediction,
and assessed against the target to determine a performance metric.

These feature selection idea is very simple; it involves taking the mean of the
responses (target) for each level (category or interval), and so amounts to a least
squares fit on a single categorical variable against a response variable, with the
categories in the continuous variables defined by intervals.

Despite its simplicity, the method has a number of advantages:

- Speed: Computing means and intervals is fast, straightforward and efficient
- Stability with respect to scale: Extreme values for continuous variables do not skew predictions as they would in many models
- Comparability between continuous and categorical variables.
- Accommodation of non-linearities.
- Does not require encoding categorical variables into numbers.

The methods has as well some limitations. First, the selection of the number of intervals
as well as the threshold are arbitrary. And also, rare categories and
very skewed variables will cause over-fitting or the model to raise an error if NAN
are accidentally introduced.

**Example**

Let's see how to use this method to select variables in the Titanic dataset. We choose
this data because it has a mix of numerical and categorical variables.

Let's go ahead and load the data and separate it into train and test:

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

Now, we set up :class:`SelectByTargetMeanPerformance()`. We will examine the roc-auc
using 2 fold cross-validation. We will separate numerical variables into
equal frequency intervals. And we will retain those variables where the roc-auc is bigger
than 0.6.

.. code:: python

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

With `fit()` the transformer:

- replaces categories by the target mean
- sorts numerical variables into equal frequency bins
- replaces bins by the target mean
- using the target mean encoded variables returns the roc-auc
- selects features which roc-auc >0.6

.. code:: python

    # find important features
    sel.fit(X_train, y_train)

The transformer stores the categorical variables identified in the data:

.. code:: python

    sel.variables_categorical_

.. code:: python

    ['pclass', 'sex', 'sibsp', 'parch', 'cabin', 'embarked']

The transformer also stores the numerical variables:

.. code:: python

    sel.variables_numerical_

.. code:: python

    ['age', 'fare']

:class:`SelectByTargetMeanPerformance()` also stores the roc-auc per feature:

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

And the features that will be dropped from the data:

.. code:: python

    sel.features_to_drop_

.. code:: python

    ['age', 'sibsp', 'parch', 'embarked']

With `transform()` we can go ahead and drop the features:

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

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
