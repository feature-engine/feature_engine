.. feature_engine documentation master file, created by
   sphinx-quickstart on Wed Jan 10 14:43:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Feature-engine: A Python library for Feature Engineering for Machine Learning
=============================================================================
.. figure::  images/FeatureEngine.png
   :align:   center

   Feature-engine rocks!

Feature-engine is a Python library with multiple transformers to engineer features for
use in machine learning models. Feature-engine preserves Scikit-learn functionality with
methods `fit()` and `transform()` to learn parameters from and then transform the data.

Feature-engine includes transformers for:

- Missing data imputation
- Categorical variable encoding
- Discretisation
- Variable transformation
- Outlier capping or removal
- Variable creation
- Variable selection

Feature-engine allows you to select the variables you want to transform within each
transformer. This way, different engineering procedures can be easily applied to
different feature subsets.

Feature-engine transformers can be assembled within the Scikit-learn pipeline,
therefore making it possible to save and deploy one single object (.pkl) with the
entire machine learning pipeline. That is, one object with the entire sequence of
variable transformations to leave the raw data ready to be consumed by a machine
learning algorithm, and the machine learning model at the back. Check the **quickstart**
for an example.

**Would you like to know more about what is unique about Feature-engine?**

This article provides a nice summary:

- `Feature-engine: A new open source Python package for feature engineering <https://trainindata.medium.com/feature-engine-a-new-open-source-python-package-for-feature-engineering-29a0ab88ea7c>`_.


Installation
------------

Feature-engine is a Python 3 package and works well with 3.6 or later. Earlier versions
have not been tested. The simplest way to install Feature-engine is from PyPI with pip:

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
---------------------------------------------------

- `Website <https://www.trainindata.com/feature-engine>`_.
- `Feature Engineering for Machine Learning <https://www.udemy.com/course/feature-engineering-for-machine-learning/?referralCode=A855148E05283015CF06>`_, Online Course.
- `Feature Selection for Machine Learning <https://www.udemy.com/course/feature-selection-for-machine-learning/?referralCode=186501DF5D93F48C4F71>`_, Online Course.
- `Python Feature Engineering Cookbook <https://www.packtpub.com/data/python-feature-engineering-cookbook>`_.
- `Feature-engine: A new open-source Python package for feature engineering <https://trainindata.medium.com/feature-engine-a-new-open-source-python-package-for-feature-engineering-29a0ab88ea7c/>`_.
- `Practical Code Implementations of Feature Engineering for Machine Learning with Python <https://towardsdatascience.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python-f13b953d4bcd>`_.

En Español:

- `Ingeniería de variables para machine learning <https://www.udemy.com/course/ingenieria-de-variables-para-machine-learning/?referralCode=CE398C784F17BD87482C>`_, Curso Online.
- `Ingeniería de variables, MachinLenin <https://www.youtube.com/watch?v=NhCxOOoFXds>`_, charla online.

More resources in the **Learning Resources** sections on the navigation panel on the
left.


Feature-engine's Transformers
-----------------------------
Missing Data Imputation: Imputers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`imputation/MeanMedianImputer`: replaces missing data in numerical variables by the mean or median
- :doc:`imputation/ArbitraryNumberImputer`: replaces missing data in numerical variables by an arbitrary value
- :doc:`imputation/EndTailImputer`: replaces missing data in numerical variables by numbers at the distribution tails
- :doc:`imputation/CategoricalImputer`: replaces missing data in categorical variables with the string 'Missing' or by the most frequent category
- :doc:`imputation/RandomSampleImputer`: replaces missing data with random samples of the variable
- :doc:`imputation/AddMissingIndicator`: adds a binary missing indicator to flag observations with missing data
- :doc:`imputation/DropMissingData`: removes rows containing NA values from dataframe

Categorical Variable Encoders: Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`encoding/OneHotEncoder`: performs one hot encoding, optional: of popular categories
- :doc:`encoding/CountFrequencyEncoder`: replaces categories by observation count or percentage
- :doc:`encoding/OrdinalEncoder`: replaces categories by numbers arbitrarily or ordered by target
- :doc:`encoding/MeanEncoder`: replaces categories by the target mean
- :doc:`encoding/WoEEncoder`: replaces categories by the weight of evidence
- :doc:`encoding/PRatioEncoder`: replaces categories by a ratio of probabilities
- :doc:`encoding/DecisionTreeEncoder`: replaces categories by predictions of a decision tree
- :doc:`encoding/RareLabelEncoder`: groups infrequent categories

Numerical Variable Transformation: Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`transformation/LogTransformer`: performs logarithmic transformation of numerical variables
- :doc:`transformation/ReciprocalTransformer`: performs reciprocal transformation of numerical variables
- :doc:`transformation/PowerTransformer`: performs power transformation of numerical variables
- :doc:`transformation/BoxCoxTransformer`: performs Box-Cox transformation of numerical variables
- :doc:`transformation/YeoJohnsonTransformer`: performs Yeo-Johnson transformation of numerical variables

Variable Discretisation: Discretisers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`discretisation/ArbitraryDiscretiser`: sorts variable into intervals arbitrarily defined by the user
- :doc:`discretisation/EqualFrequencyDiscretiser`: sorts variable into equal frequency intervals
- :doc:`discretisation/EqualWidthDiscretiser`: sorts variable into equal size contiguous intervals
- :doc:`discretisation/DecisionTreeDiscretiser`: uses decision trees to create finite variables

Outlier Capping or Removal
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`outliers/ArbitraryOutlierCapper`: caps maximum and minimum values at user defined values
-  :doc:`outliers/Winsorizer`: caps maximum or minimum values using statistical parameters
-  :doc:`outliers/OutlierTrimmer`: removes outliers from the dataset

Scikit-learn Wrapper:
~~~~~~~~~~~~~~~~~~~~~

-  :doc:`wrappers/Wrapper`: executes Scikit-learn various transformers only on the selected subset of features

Mathematical Combination:
~~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`creation/MathematicalCombination`: creates new variables by combining features with mathematical operations
-  :doc:`creation/CombineWithReferenceFeature`: creates variables with reference features through mathematical operations

Feature Selection:
~~~~~~~~~~~~~~~~~~

- :doc:`selection/DropFeatures`: drops a subset of variables from a dataframe
- :doc:`selection/DropConstantFeatures`: drops constant and quasi-constant variables from a dataframe
- :doc:`selection/DropDuplicateFeatures`: drops duplicated variables from a dataframe
- :doc:`selection/DropCorrelatedFeatures`: drops correlated variables from a dataframe
- :doc:`selection/SmartCorrelatedSelection`: selects best feature from correlated group
- :doc:`selection/SelectByShuffling`: selects features by evaluating model performance after feature shuffling
- :doc:`selection/SelectBySingleFeaturePerformance`: selects features based on their performance on univariate estimators
- :doc:`selection/SelectByTargetMeanPerformance`: selects features based on target mean encoding performance
- :doc:`selection/RecursiveFeatureElimination`: selects features recursively, by evaluating model performance
- :doc:`selection/RecursiveFeatureAddition`: selects features recursively, by evaluating model performance


Getting Help
------------

Can't get something to work? Here are places where you can find help.

1. The docs (you're here!).
2. `Stack Overflow <https://stackoverflow.com/questions/tagged/feature-engine>`_. If you ask a question, please tag it with "feature-engine".
3. If you are enrolled in the `Feature Engineering for Machine Learning course in Udemy <https://www.udemy.com/course/feature-engineering-for-machine-learning/?referralCode=A855148E05283015CF06>`_ , post a question in a relevant section.
4. If you are enrolled in the `Feature Selection for Machine Learning course in Udemy <https://www.udemy.com/course/feature-selection-for-machine-learning/?referralCode=186501DF5D93F48C4F71>`_ , post a question in a relevant section.
5. Join our `mailing list <https://groups.google.com/d/forum/feature-engine>`_.
6. Ask a question in the repo by filing an `issue <https://github.com/solegalli/feature_engine/issues/>`_.


Found a Bug or have a suggestion?
---------------------------------

Check if there's already an open `issue <https://github.com/solegalli/feature_engine/issues/>`_
on the topic. If not, open a new `issue <https://github.com/solegalli/feature_engine/issues/>`_
with your bug report, suggestion or new feature request.

Contributing
------------

Interested in contributing to Feature-engine? That is great news!

Feature-engine is a welcoming and inclusive project and it would be great to have you
on board. We follow the
`Python Software Foundation Code of Conduct <http://www.python.org/psf/codeofconduct/>`_.

Regardless of your skill level you can help us. We appreciate bug reports, user testing,
feature requests, bug fixes, addition of tests, product enhancements, and documentation
improvements. We also appreciate blogs about Feature-engine. If you happen to have one,
let us know!

For more details on how to contribute check the contributing page. Click on the
"Contributing" link on the left of this page.


Open Source
-----------

Feature-engine's `license <https://github.com/solegalli/feature_engine/blob/master/LICENSE.md>`_
is an open source BSD 3-Clause.

Feature-engine is hosted on `GitHub <https://github.com/solegalli/feature_engine/>`_.
The `issues <https://github.com/solegalli/feature_engine/issues/>`_ and
`pull requests <https://github.com/solegalli/feature_engine/pulls>`_ are tracked there.



.. toctree::
   :maxdepth: 1
   :caption: Table of Contents
   
   quickstart
   installation
   getting_help
   about
   datasets

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   imputation/index
   encoding/index
   transformation/index
   discretisation/index
   outliers/index
   creation/index
   selection/index
   wrappers/index

.. toctree::
   :maxdepth: 1
   :caption: Learning Resources

   tutorials
   howto
   books
   courses
   blogs

.. toctree::
   :maxdepth: 1
   :caption: Contribute

   contribute/index
   code_of_conduct
   governance

.. toctree::
   :maxdepth: 1
   :caption: Releases

   whats_new/index
