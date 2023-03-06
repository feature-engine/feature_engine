.. _probe_features:

.. py:currentmodule:: feature_engine.selection

ProbeFeatureSelection
=====================

ProbeFeatureSelection() generates one or more probe features based on the
user-selected parameters.

The class derives the feature importance score for each variable and probe feature.
In the case of there being more than one probe feature, the average feature importance
score of all the probe features is used.

The class ranks the features based on their importance and eliminates the features
that have a lower feature importance score than the probe feature(s).

This selection method was published in the Journal of Machine Learning Research in 2003.

One of the primary goals of feature selection is to remove noise from the dataset. A
randomly generated variable, i.e., probe feature, inherently possesses a high level of
noise. Consequently, any variable that demonstrates less importance than a probe feature
is assumed to be noise and can be discarded from the dataset.

When initiating the ProbeFeatureSelection() class, the user has the option of selecting
which distribution is to be assumed to create the probe feature(s) and the number of
probe features to be created. The possible distributions are 'normal', 'binary', 'uniform',
or 'all'. 'all' creates 1 or more probe features comprised of each distribution type,
i.e., normal, binomial, and uniform.

Example
-------
Let's see how to use this transformer to select variables from UC Irvine's Breast Cancer
Wisconsin (Diagnostic) dataset, which can be found `here`_. We will use Scikit-learn to load
the dataset. This dataset concerns breast cancer diagnoses. The target variable is binary, i.e.,
malignant or benign.

The data is solely comprised of numerical data.

.. _here: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Let's import the required libraries and classes:

.. code:: python

    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from feature_engine.selection import SelectByInformationValue

Let's now load the cancer diagnostic data:

.. code:: python

    cancer_X, cancer_y = load_breast_cancer(return_X_y=True, as_frame=True)

Let's check the shape of `cancer_X`:

.. code:: python

    print(cancer_X.shape)


We see that the dataset is comprised of 569 observations and 30 features:

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


Now, we set up :class:`ProbeFeatureSelection()`. We will pass  `RandomForestClassifier()`
as the :code:`estimator`. We will use `precision` as the :code:`scoring` parameter
and `5` as :code:`cv` parameter, both parameters to be be used in the cross validation.
We will assume `1` for the :code:`n_probes` parameter and `normal` as the :code:`distribution`,
both parameters to be used when creating the probe feature.


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
    - calculates the feature importance score for each variable, including probe features
    - if there are multiple probe features, calculate the average importance score
    - identifies features to drop because their importance scores are less than that of the probe feature(s)


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

With :code:`transform()`, we can go ahead and drop the six features with feature importance score
less than `gaussian_probe_0` variable:

.. code:: python

    Xtr = sel.transform(X_test)

    Xtr.shape

    (114, 24)


And, finally, we can also obtain the names of the features in the final transformed dataset:

.. code:: python

    sel.get_feature_names_out()

    ['mean radius',
     'mean texture',
     'mean perimeter',
     'mean area',
     'mean smoothness',
     'mean compactness',
     'mean concavity',
     'mean concave points',
     'radius error',
     'perimeter error',
     'area error',
     'compactness error',
     'concavity error',
     'symmetry error',
     'worst radius',
     'worst texture',
     'worst perimeter',
     'worst area',
     'worst smoothness',
     'worst compactness',
     'worst concavity',
     'worst concave points',
     'worst symmetry',
     'worst fractal dimension']


