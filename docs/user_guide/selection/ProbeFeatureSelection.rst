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

Let's now load and prepare the credit approval data:

.. code:: python
