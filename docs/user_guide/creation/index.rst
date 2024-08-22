.. -*- mode: rst -*-

Feature Creation
================

Feature creation, is a common step during data preprocessing, and consists of constructing new
variables from the dataset’s original features. By combining two or more variables, we develop
new features that can improve the performance of a machine learning model, capture additional
information or relationships among variables, or simply make more sense within the domain we
are working on.

One of the most common feature creation methods in data science is `one-hot
encoding <https://www.blog.trainindata.com/one-hot-encoding-categorical-variables/>`_, which
is a feature engineering technique used to transform a categorical feature into multiple
binary variables that represent each category.

Another common feature extraction procedure consist of creating new features from past
values of time series data, for example through the use of lags and windows.

In general, creating features requires a dose of domain knowledge and significant time
invested in analyzing the raw data, including evaluating the relationship between the independent or
predictor variables and the dependent or target variable in the dataset.

Feature creation can be one of the more creative aspects of feature engineering, and the new
features can help improve a predictive model’s performance.

Lastly, a data scientist should be mindful that creating new features may increase the dimensionality
of the dataset quite dramatically. For example, one hot encoding of highly cardinal categorical
features results in lots of binary variables, and so does polynomial combinations of high powers.
This may have downstream effects depending on the machine learning algorithm being used. For example,
decision trees are known for not being able to cope with huge number of features.

Creating New Features with Feature-engine
-----------------------------------------

Feature-engine has several transformers that create and add new features to the dataset. One of
the most popular ones is the `OneHotEncoder <https://feature-engine.trainindata.com/en/latest/user_guide/encoding/OneHotEncoder.html>`_
that creates dummy variables from categorical features.

With Feature-engine we can also create new features from time series data through lags and windows by using
`LagFeatures <https://feature-engine.trainindata.com/en/latest/user_guide/timeseries/forecasting/LagFeatures.html>`_
or `WindowFeatures <https://feature-engine.trainindata.com/en/latest/user_guide/timeseries/forecasting/WindowFeatures.html>`_.

Feature-engine’s creation module, supports transformers that create and add new features to a pandas
dataframe by either combining existing features through different mathematical or statistical operations,
or through feature transformations. These transformers operate with numerical variables, that is, those
with integer and float data types.

Summary of Feature-engine’s feature-creation transformers:

- **CyclicalFeatures** - Creates two new features per variable by applying the trigonometric operations sine and cosine to the original feature.

- **MathFeatures** - Combines a set of features into new variables by applying basic mathematical functions like the sum, mean, maximum or standard deviation.

- **RelativeFeatures** - Utilizes basic mathematical functions between a group of variables and one or more reference features, appending the new features to the pandas dataframe.

- **DecisionTreeFeatures** - Creates new features as the output of decision trees trained on 1 or more feature combinations.

Feature creation module
-----------------------

.. toctree::
   :maxdepth: 1

   CyclicalFeatures
   MathFeatures
   RelativeFeatures
   DecisionTreeFeatures

Feature-engine in Practice
--------------------------

Here, you'll get a taste of the transformers from the feature creation module from Feature-engine.
We'll use the wine quality dataset. The dataset is comprised of 11 features, including `alcohol`,
`ash`, and ``flavonoids``, and has `quality` as its target variable.

Through exploratory data analysis and our domain knowledge which includes real-world
experimentation, i.e., drinking various brands/types of wine, we believe that we can
create better features to train our algorithm by combining original features with various
mathematical operations.

Let's load the dataset from Scikit-learn.

.. code:: python

    import pandas as pd
    from sklearn.datasets import load_wine
    from feature_engine.creation import RelativeFeatures, MathFeatures

    X, y = load_wine(return_X_y=True, as_frame=True)
    print(X.head())

Below we see the wine quality dataset:

.. code:: python

       alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \
    0    14.23        1.71  2.43               15.6      127.0           2.80
    1    13.20        1.78  2.14               11.2      100.0           2.65
    2    13.16        2.36  2.67               18.6      101.0           2.80
    3    14.37        1.95  2.50               16.8      113.0           3.85
    4    13.24        2.59  2.87               21.0      118.0           2.80

       flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \
    0        3.06                  0.28             2.29             5.64  1.04
    1        2.76                  0.26             1.28             4.38  1.05
    2        3.24                  0.30             2.81             5.68  1.03
    3        3.49                  0.24             2.18             7.80  0.86
    4        2.69                  0.39             1.82             4.32  1.04

       od280/od315_of_diluted_wines  proline
    0                          3.92   1065.0
    1                          3.40   1050.0
    2                          3.17   1185.0
    3                          3.45   1480.0
    4                          2.93    735.0


Now, we create a new feature by removing non-flavonoid phenols from the total phenols to
obtain the phenols that are not flavonoid.

.. code:: python

    rf = RelativeFeatures(
        variables=["total_phenols"],
        reference=["nonflavanoid_phenols"],
        func=["sub"],
    )

    rf.fit(X)
    X_tr = rf.transform(X)

    print(X_tr.head())

We see the new feature and its data points at the right of the pandas dataframe:

.. code:: python

       alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \
    0    14.23        1.71  2.43               15.6      127.0           2.80
    1    13.20        1.78  2.14               11.2      100.0           2.65
    2    13.16        2.36  2.67               18.6      101.0           2.80
    3    14.37        1.95  2.50               16.8      113.0           3.85
    4    13.24        2.59  2.87               21.0      118.0           2.80

       flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \
    0        3.06                  0.28             2.29             5.64  1.04
    1        2.76                  0.26             1.28             4.38  1.05
    2        3.24                  0.30             2.81             5.68  1.03
    3        3.49                  0.24             2.18             7.80  0.86
    4        2.69                  0.39             1.82             4.32  1.04

       od280/od315_of_diluted_wines  proline  \
    0                          3.92   1065.0
    1                          3.40   1050.0
    2                          3.17   1185.0
    3                          3.45   1480.0
    4                          2.93    735.0

       total_phenols_sub_nonflavanoid_phenols
    0                                    2.52
    1                                    2.39
    2                                    2.50
    3                                    3.61
    4                                    2.41


Let's now create new features by combining a subset of 3 existing variables:

.. code:: python

    mf = MathFeatures(
        variables=["flavanoids", "proanthocyanins", "proline"],
        func=["sum", "mean"],
    )

    mf.fit(X_tr)
    X_tr = mf.transform(X_tr)

    print(X_tr.head())

We see the new features at the right of the resulting pandas dataframe:

.. code:: python

       alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \
    0    14.23        1.71  2.43               15.6      127.0           2.80
    1    13.20        1.78  2.14               11.2      100.0           2.65
    2    13.16        2.36  2.67               18.6      101.0           2.80
    3    14.37        1.95  2.50               16.8      113.0           3.85
    4    13.24        2.59  2.87               21.0      118.0           2.80

       flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \
    0        3.06                  0.28             2.29             5.64  1.04
    1        2.76                  0.26             1.28             4.38  1.05
    2        3.24                  0.30             2.81             5.68  1.03
    3        3.49                  0.24             2.18             7.80  0.86
    4        2.69                  0.39             1.82             4.32  1.04

       od280/od315_of_diluted_wines  proline  \
    0                          3.92   1065.0
    1                          3.40   1050.0
    2                          3.17   1185.0
    3                          3.45   1480.0
    4                          2.93    735.0

       total_phenols_sub_nonflavanoid_phenols  \
    0                                    2.52
    1                                    2.39
    2                                    2.50
    3                                    3.61
    4                                    2.41

       sum_flavanoids_proanthocyanins_proline  \
    0                                 1070.35
    1                                 1054.04
    2                                 1191.05
    3                                 1485.67
    4                                  739.51

       mean_flavanoids_proanthocyanins_proline
    0                               356.783333
    1                               351.346667
    2                               397.016667
    3                               495.223333
    4                               246.503333


In the above examples, we used `RelativeFeature()` and `MathFeatures` to perform automated feature
engineering on the input data by applying the transformations defined in the `func` parameter on
the features identified in `variables`  and ``reference`` parameters.

The original and new features can now be used to train a regression model, or a multiclass
classification algorithm, to predict the `quality` of the wine.

Summary
-------

Through feature engineering and feature creation, we can optimize the machine learning algorithm's
learning process and improve its performance metrics.

We'd strongly recommend the creation of features based on domain knowledge, exploratory data
analysis and thorough data mining. We also understand that this is not always possible, particularly
with big datasets and limited time allocated to each project. In this situation, we can combine
the creation of features with feature selection procedures to let machine learning algorithms
select what works best for them.

Good luck with your models!


Tutorials, books and courses
----------------------------

For tutorials about this and other feature engineering for machine learning methods check out
our online course:

.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

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

.. figure::  ../../images/cookbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587

   Python Feature Engineering Cookbook

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


Transformers in other Libraries
-------------------------------

Check also the following transformer from Scikit-learn:

* `PolynomialFeatures <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_
* `SplineTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html>`_
