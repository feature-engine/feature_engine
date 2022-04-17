.. feature_engine documentation master file, created by
   sphinx-quickstart on Wed Jan 10 14:43:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Feature-engine
==============

A Python library for Feature Engineering and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure::  images/logo/FeatureEngine.png
   :align:   center

   **Feature-engine rocks!**

Feature-engine is a Python library with multiple transformers to engineer and select
features to use in machine learning models. Feature-engine preserves Scikit-learn
functionality with methods `fit()` and `transform()` to learn parameters from and then
transform the data.

Feature-engine includes transformers for:

- Missing data imputation
- Categorical encoding
- Discretisation
- Outlier capping or removal
- Variable transformation
- Variable creation
- Variable selection
- Datetime features
- Time series
- Preprocessing

Feature-engine allows you to select the variables you want to transform **within** each
transformer. This way, different engineering procedures can be easily applied to
different feature subsets.

Feature-engine transformers can be assembled within the Scikit-learn pipeline,
therefore making it possible to save and deploy one single object (.pkl) with the
entire machine learning pipeline. Check :ref:`**Quick Start** <quick_start>` for an
example.

What is unique about Feature-engine?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following characteristics make Feature-engine unique:

- Feature-engine contains the most exhaustive battery of feature engineering transformations.
- Feature-engine can transform a specific group of variables in the dataframe.
- Feature-engine returns dataframes, hence suitable for data exploration and model deployment.
- Feature-engine is compatible with the Scikit-learn pipeline.
- Feature-engine automatically recognizes numerical, categorical and datetime variables.
- Feature-engine alerts you if a transformation is not possible, e.g., if applying logarithm to negative variables or divisions by 0.

If you want to know more about what makes Feature-engine unique, check this
`article <https://trainindata.medium.com/feature-engine-a-new-open-source-python-package-for-feature-engineering-29a0ab88ea7c>`_.


Installation
~~~~~~~~~~~~

Feature-engine is a Python 3 package and works well with 3.7 or later. Earlier versions
are not compatible with the latest versions of Python numerical computing libraries.

The simplest way to install Feature-engine is from PyPI with pip:

.. code-block:: bash

    $ pip install feature-engine

Note, you can also install it with a _ as follows:

.. code-block:: bash

    $ pip install feature_engine

Feature-engine is an active project and routinely publishes new releases. To upgrade
Feature-engine to the latest version, use pip like this:

.. code-block:: bash

    $ pip install -U feature-engine

If you’re using Anaconda, you can install the
`Anaconda Feature-engine package <https://anaconda.org/conda-forge/feature_engine>`_:

.. code-block:: bash

    $ conda install -c conda-forge feature_engine


Feature-engine features in the following resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Feature Engineering for Machine Learning <https://courses.trainindata.com/p/feature-engineering-for-machine-learning>`_, Online Course.
- `Feature Selection for Machine Learning <https://courses.trainindata.com/p/feature-selection-for-machine-learning>`_, Online Course.
- `Feature Engineering for Time Series Forecasting <https://www.courses.trainindata.com/p/feature-engineering-for-forecasting>`_, Online Course.
- `Python Feature Engineering Cookbook <https://packt.link/python>`_.
- `Feature-engine: A new open-source Python package for feature engineering <https://trainindata.medium.com/feature-engine-a-new-open-source-python-package-for-feature-engineering-29a0ab88ea7c/>`_.
- `Practical Code Implementations of Feature Engineering for Machine Learning with Python <https://towardsdatascience.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python-f13b953d4bcd>`_.

More learning resources in the :ref:`**Learning Resources** <learning_resources>`.


Feature-engine's Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature-engine hosts the following groups of transformers:

Missing Data Imputation: Imputers
---------------------------------

- :doc:`api_doc/imputation/MeanMedianImputer`: replaces missing data in numerical variables by the mean or median
- :doc:`api_doc/imputation/ArbitraryNumberImputer`: replaces missing data in numerical variables by an arbitrary number
- :doc:`api_doc/imputation/EndTailImputer`: replaces missing data in numerical variables by numbers at the distribution tails
- :doc:`api_doc/imputation/CategoricalImputer`: replaces missing data with an arbitrary string or by the most frequent category
- :doc:`api_doc/imputation/RandomSampleImputer`: replaces missing data by random sampling observations from the variable
- :doc:`api_doc/imputation/AddMissingIndicator`: adds a binary missing indicator to flag observations with missing data
- :doc:`api_doc/imputation/DropMissingData`: removes observations (rows) containing missing values from dataframe

Categorical Encoders: Encoders
------------------------------

- :doc:`api_doc/encoding/OneHotEncoder`: performs one hot encoding, optional: of popular categories
- :doc:`api_doc/encoding/CountFrequencyEncoder`: replaces categories by the observation count or percentage
- :doc:`api_doc/encoding/OrdinalEncoder`: replaces categories by numbers arbitrarily or ordered by target
- :doc:`api_doc/encoding/MeanEncoder`: replaces categories by the target mean
- :doc:`api_doc/encoding/WoEEncoder`: replaces categories by the weight of evidence
- :doc:`api_doc/encoding/PRatioEncoder`: replaces categories by a ratio of probabilities
- :doc:`api_doc/encoding/DecisionTreeEncoder`: replaces categories by predictions of a decision tree
- :doc:`api_doc/encoding/RareLabelEncoder`: groups infrequent categories

Variable Discretisation: Discretisers
-------------------------------------

- :doc:`api_doc/discretisation/ArbitraryDiscretiser`: sorts variable into intervals defined by the user
- :doc:`api_doc/discretisation/EqualFrequencyDiscretiser`: sorts variable into equal frequency intervals
- :doc:`api_doc/discretisation/EqualWidthDiscretiser`: sorts variable into equal width intervals
- :doc:`api_doc/discretisation/DecisionTreeDiscretiser`: uses decision trees to create finite variables

Outlier Capping or Removal
--------------------------

-  :doc:`api_doc/outliers/ArbitraryOutlierCapper`: caps maximum and minimum values at user defined values
-  :doc:`api_doc/outliers/Winsorizer`: caps maximum or minimum values using statistical parameters
-  :doc:`api_doc/outliers/OutlierTrimmer`: removes outliers from the dataset

Numerical Transformation: Transformers
--------------------------------------

- :doc:`api_doc/transformation/LogTransformer`: performs logarithmic transformation of numerical variables
- :doc:`api_doc/transformation/LogCpTransformer`: performs logarithmic transformation after adding a constant value
- :doc:`api_doc/transformation/ReciprocalTransformer`: performs reciprocal transformation of numerical variables
- :doc:`api_doc/transformation/PowerTransformer`: performs power transformation of numerical variables
- :doc:`api_doc/transformation/BoxCoxTransformer`: performs Box-Cox transformation of numerical variables
- :doc:`api_doc/transformation/YeoJohnsonTransformer`: performs Yeo-Johnson transformation of numerical variables

Feature Creation:
-----------------

-  :doc:`api_doc/creation/MathFeatures`: creates new variables by combining features with mathematical operations
-  :doc:`api_doc/creation/RelativeFeatures`: combines variables with reference features
-  :doc:`api_doc/creation/CyclicalFeatures`: creates variables using sine and cosine, suitable for cyclical features

Feature Selection:
------------------

- :doc:`api_doc/selection/DropFeatures`: drops an arbitrary subset of variables from a dataframe
- :doc:`api_doc/selection/DropConstantFeatures`: drops constant and quasi-constant variables from a dataframe
- :doc:`api_doc/selection/DropDuplicateFeatures`: drops duplicated variables from a dataframe
- :doc:`api_doc/selection/DropCorrelatedFeatures`: drops correlated variables from a dataframe
- :doc:`api_doc/selection/SmartCorrelatedSelection`: selects best features from correlated groups
- :doc:`api_doc/selection/DropHighPSIFeatures`: selects features based on the Population Stability Index (PSI)
- :doc:`api_doc/selection/SelectByShuffling`: selects features by evaluating model performance after feature shuffling
- :doc:`api_doc/selection/SelectBySingleFeaturePerformance`: selects features based on their performance on univariate estimators
- :doc:`api_doc/selection/SelectByTargetMeanPerformance`: selects features based on target mean encoding performance
- :doc:`api_doc/selection/RecursiveFeatureElimination`: selects features recursively, by evaluating model performance
- :doc:`api_doc/selection/RecursiveFeatureAddition`: selects features recursively, by evaluating model performance

Datetime:
---------

- :doc:`api_doc/datetime/DatetimeFeatures`: extract features from datetime variables

Forecasting:
------------

- :doc:`api_doc/timeseries/forecasting/LagFeatures`: extract lag features
- :doc:`api_doc/timeseries/forecasting/WindowFeatures`: create window features
- :doc:`api_doc/timeseries/forecasting/ExpandingWindowFeatures`: create expanding window features

Preprocessing:
--------------

- :doc:`api_doc/preprocessing/MatchVariables`: ensures that columns in test set match those in train set

Scikit-learn Wrapper:
---------------------

-  :doc:`api_doc/wrappers/Wrapper`: applies Scikit-learn transformers to a selected subset of features


Getting Help
~~~~~~~~~~~~

Can't get something to work? Here are places where you can find help.

1. The :ref:`**User Guide** <user_guide>` in the docs.
2. `Stack Overflow <https://stackoverflow.com/search?q=feature_engine>`_. If you ask a question, please mention "feature_engine" in it.
3. If you are enrolled in the `Feature Engineering for Machine Learning course <https://courses.trainindata.com/p/feature-engineering-for-machine-learning>`_ , post a question in a relevant section.
4. If you are enrolled in the `Feature Selection for Machine Learning course <https://courses.trainindata.com/p/feature-selection-for-machine-learning>`_ , post a question in a relevant section.
5. Join our `gitter community <https://gitter.im/feature_engine/community>`_. You an ask questions here as well.
6. Ask a question in the repo by filing an `issue <https://github.com/feature-engine/feature_engine/issues/>`_ (check before if there is already a similar issue created :) ).


Contributing
~~~~~~~~~~~~

Interested in contributing to Feature-engine? That is great news!

Feature-engine is a welcoming and inclusive project and we would be delighted to have you
on board. We follow the
`Python Software Foundation Code of Conduct <http://www.python.org/psf/codeofconduct/>`_.

Regardless of your skill level you can help us. We appreciate bug reports, user testing,
feature requests, bug fixes, addition of tests, product enhancements, and documentation
improvements. We also appreciate blogs about Feature-engine. If you happen to have one,
let us know!

For more details on how to contribute check the contributing page. Click on the
:ref:`**Contribute** <contribute>` guide.

Sponsor us
~~~~~~~~~~

Support Feature-engine financially via
`Github Sponsors <https://github.com/sponsors/feature-engine>`_ and help further our
mission to democratize machine learning tools through open-source software.

Open Source
~~~~~~~~~~~

Feature-engine's `license <https://github.com/feature-engine/feature_engine/blob/master/LICENSE.md>`_
is an open source BSD 3-Clause.

Feature-engine is hosted on `GitHub <https://github.com/feature-engine/feature_engine/>`_.
The `issues <https://github.com/feature-engine/feature_engine/issues/>`_ and
`pull requests <https://github.com/feature-engine/feature_engine/pulls>`_ are tracked there.


Table of Contents
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   quickstart/index
   user_guide/index
   api_doc/index
   resources/index
   contribute/index
   about/index
   whats_new/index
   donate
