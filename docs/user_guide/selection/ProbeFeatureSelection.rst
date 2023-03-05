.. _probe_features:

.. py:currentmodule:: feature_engine.selection

ProbeFeatureSelection
=====================

ProbeFeatureSelection() generates one or more probe features based on the
user-selected distribution.

The class derives the feature importance for each variable and probe feature.
In the case of there being more than one probe feature, ProbeFeatureSelection()
calculates the average feature importance of all the probe features.

The class ranks the features based on their importance and eliminates the features
that have a lower importance than the probe features.

This selection method was published in the Journal of Machine Learning Research in 2003.

One of the primary goals of feature selection is to remove noise from the dataset. A
randomly generated variable, i.e., probe feature, inherently possesses a high level of
noise. Consequently, any variable that demonstrates less importance than a probe feature
is assumed to be noise and can be discarded from the dataset.

When initiated the ProbeFeatureSelection() class, the user has the option of selecting
which distribution is to be assumed when create the probe feature(s) and the number of
probe features to be created. The possible distributions are 'normal', 'binary', 'uniform',
or 'all'. 'all' creates 1 or more probe features comprised of each distribution type,
i.e., normal, binomial, and uniform.

Example
-------
Let's see how to use this transformer to select variables from UC Irvine's Breast Cancer
Wisconsin (Diagnostic) dataset, which can be round `here`_. We will use Scikit-learn to load
the dataset. This dataset concerns breast cancer diagnoses. The target variable is binary, i.e.,
malignant or benign.

The data is soley comprised of numerical data.

.. _here: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Let's import the required libraries and classes:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from feature_engine.selection import SelectByInformationValue

Let's now load the cancer diagnostic data:

.. code:: python

    cancer_X, cancer_y = load_breast_cancer(return_X_y=True, as_frame=True)

Let's check the shape of `cancer_X`:

.. code:: python

    print(cancer_X.shape)


We see that the dataset comprised of 569 observations and 30 features:

.. code:: python

    (569, 30)


Let's now split the data into train and test sets:

.. code:: python


    # separate train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        cancer_X,
        cancer_y,
        test_size=0.2,
        random_state=3
    )

    X_train.shape, X_test.shape

We see the size of the datasets below. Note that there are 30 features in both the
training and test sets.

.. code:: python

    ((455, 30), (114, 30))

Now, we set up :class:`SelectByInformationValue()`. We will pass six categorical
variables to the parameter :code:`variables`. We will set the parameter :code:`threshold`
to `0.2`. We see from the above mentioned table that an IV score of 0.2 signifies medium
predictive power.


Now, we set up :class:`ProbeFeatureSelection()`. We will pass the `RandomForestClassifier()`
as the estimator :code:`estimator`. We will use `precision` as the :code:`scoring` parameter
and `5` as :code:`cv` parameter, both to be be used in the cross validation. We will assume
`1` for the :code:`n_probes` parameter and `normal` as the :code:`distribution`, both
parameters to be used when creating the probe feature.


.. code:: python

    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(),
        variables=None,
        scoring="precision",
        n_probes=1,
        distribution="normal",
        cv=5,
        random_state=150,
        confirm_variables=False
    )

    sel.fit(X_train, y_train)

With :code:`fit()`, the transformer:

    - creates `n_probes` number of probe features using provided distribution(s)
    - uses cross-validation to fit the provided estimator
    - calculates the feature importance for each variable
    - if there are multiple probe features, calculate the average importance
    - identifies features to drop because their importances are less than that of the probe feature


In the attribute :code:`probe_features`, we find the pseudo-randomly generated variable(s):

.. code:: python

    sel.probe_features_.head()

           gaussian_probe_0
    0         -0.694150
    1          1.171840
    2          1.074892
    3          1.698733
    4          0.498702


The attribute :code:`feature_importances_` shows each variable's feature importance:

.. code:: python

    sel.feature_importances.head()

    mean radius        0.058463
    mean texture       0.011953
    mean perimeter     0.069516
    mean area          0.050947
    mean smoothness    0.004974

In the attribute :code:`features_to_drop_`, we find the variables that were not selected:

.. code:: python

    sel.features_to_drop_

    ['mean symmetry',
     'mean fractal dimension',
     'texture error',
     'smoothness error',
     'concave points error',
     'fractal dimension error']

We see that the :code:`features_to_drop` have feature importance scores that are less
than the probe feature's score:

.. code:: python

    vars_to_drop = sel.features_to_drop_
    vars_to_display = vars_to_drop + ["gaussian_probe_0"]

    mean symmetry              0.003698
    mean fractal dimension     0.003455
    texture error              0.003595
    smoothness error           0.003333
    concave points error       0.003548
    fractal dimension error    0.003576
    gaussian_probe_0           0.003783




