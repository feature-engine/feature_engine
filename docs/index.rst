.. feature_engine documentation master file, created by
   sphinx-quickstart on Wed Jan 10 14:43:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Feature-engine: A Python library for Feature Engineering for Machine Learning
=============================================================================
.. figure::  images/FeatureEngine.png
   :align:   center

   Feature-engine rocks!

Feature-engine is a Python library with multiple transformers to engineer features for use
in machine learning models. Feature-engine preserves Scikit-learn functionality with methods
fit() and transform() to learn parameters from and then transform the data.

Feature-engine includes transformers for:

- Missing data imputation
- Categorical variable encoding
- Discretisation
- Numerical variable transformation
- Outlier capping or removal
- Variables combination
- Variable selection

Feature-engine allows you to select the variables you want to engineer or transform within each transformer.
This way, different engineering procedures can be easily applied to different feature subsets.

Feature-engine's transformers can be assembled within the Scikit-learn pipeline, therefore making it
possible to save and deploy one single object (.pkl) with the entire machine learning pipeline. That is, with
the entire sequence of transformations to transform your raw data into data that can be fed to machine learning
algorithms.

Would you like to know more about what is unique about Feature-engine?

This article provides a nice summary:
`Feature-engine: A new open source Python package for feature engineering <https://www.trainindatablog.com/feature-engine-a-new-open-source-python-package-for-feature-engineering>`_.


Installation
------------

Feature-engine is a Python 3 package and works well with 3.6 or later. Earlier versions have not been tested.
The simplest way to install Feature-engine is from PyPI with pip, Python's preferred package installer:

.. code-block:: bash

    $ pip install feature-engine

Note, you can also install it with a _ as follows:

.. code-block:: bash

    $ pip install feature_engine

Feature-engine is an active project and routinely publishes new releases with new or updated transformers.
In order to upgrade Feature-engine to the latest version, use pip like this:

.. code-block:: bash

    $ pip install -U feature-engine

If you’re using Anaconda, you can take advantage of the conda utility to install the `Anaconda Feature-engine package <https://anaconda.org/conda-forge/feature_engine>`_:

.. code-block:: bash

    $ conda install -c conda-forge feature_engine


Feature-engine features in the following resources
---------------------------------------------------

- `Home page <https://www.trainindata.com/feature-engine>`_.
- `Feature Engineering for Machine Learning, Online Course <https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO>`_.
- `Python Feature Engineering Cookbook <https://www.packtpub.com/data/python-feature-engineering-cookbook>`_.
- `Feature-engine: A new open-source Python package for feature engineering <https://www.trainindatablog.com/feature-engine-a-new-open-source-python-package-for-feature-engineering/>`_.
- `Practical Code Implementations of Feature Engineering for Machine Learning with Python <https://www.trainindatablog.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python/>`_.

En Español:

- `Ingeniería de variables para machine learning, Curso Online <https://www.udemy.com/course/ingenieria-de-variables-para-machine-learning/?referralCode=CE398C784F17BD87482C>`_.
- `Ingeniería de variables, MachinLenin, charla online <https://www.youtube.com/watch?v=NhCxOOoFXds>`_.

More resources will be added as they appear online!

Contributing
------------

Interested in contributing to Feature-engine? That is great news!

Feature-engine is a welcoming and inclusive project and it would be great to have you on board. We follow the
`Python Software Foundation Code of Conduct <http://www.python.org/psf/codeofconduct/>`_.

Regardless of your skill level you can help us. We appreciate bug reports, user testing, feature requests, bug fixes,
addition of tests, product enhancements, and documentation improvements.

We also appreciate blogs about Feature-engine. If you happen to have one, let us know!

For more details on how to contribute check the contributing page. Click on the "Contributing" page in the
"Table of Contents" on the left of this page.

Thank you for your contributions!


Feature-engine's Transformers
-----------------------------
Missing Data Imputation: Imputers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`imputers/MeanMedianImputer`: replaces missing data in numerical variables by the mean or median
- :doc:`imputers/ArbitraryValueImputer`: replaces missing data in numerical variables by an arbitrary value
- :doc:`imputers/EndTailImputer`: replaces missing data in numerical variables by numbers at the distribution tails
- :doc:`imputers/CategoricalVariableImputer`: replaces missing data in categorical variables with the string 'Missing' or by the most frequent category
- :doc:`imputers/RandomSampleImputer`: replaces missing data with random samples of the variable
- :doc:`imputers/AddMissingIndicator`: adds a binary missing indicator to flag observations with missing data

Categorical Variable Encoders: Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`encoders/OneHotCategoricalEncoder`: performs one hot encoding, optional: of popular categories
- :doc:`encoders/CountFrequencyCategoricalEncoder`: replaces categories by observation count or percentage
- :doc:`encoders/OrdinalCategoricalEncoder`: replaces categories by numbers arbitrarily or ordered by target
- :doc:`encoders/MeanCategoricalEncoder`: replaces categories by the target mean
- :doc:`encoders/WoERatioCategoricalEncoder`: replaces categories by the weight of evidence
- :doc:`encoders/DecisionTreeCategoricalEncoder`: replaces categories by predictions of a decision tree
- :doc:`encoders/RareLabelCategoricalEncoder`: groups infrequent categories

Numerical Variable Transformation: Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`vartransformers/LogTransformer`: performs logarithmic transformation of numerical variables
- :doc:`vartransformers/ReciprocalTransformer`: performs reciprocal transformation of numerical variables
- :doc:`vartransformers/PowerTransformer`: performs power transformation of numerical variables
- :doc:`vartransformers/BoxCoxTransformer`: performs Box-Cox transformation of numerical variables
- :doc:`vartransformers/YeoJohnsonTransformer`: performs Yeo-Johnson transformation of numerical variables

Variable Discretisation: Discretisers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`discretisers/EqualFrequencyDiscretiser`: sorts variable into equal frequency intervals
- :doc:`discretisers/EqualWidthDiscretiser`: sorts variable into equal size contiguous intervals
- :doc:`discretisers/DecisionTreeDiscretiser`: uses decision trees to create finite variables
- :doc:`discretisers/UserInputDiscretiser`: allows the user to arbitrarily define the intervals


Outlier Capping or Removal
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`outliercappers/Winsorizer`: caps maximum or minimum values using statistical parameters
-  :doc:`outliercappers/ArbitraryOutlierCapper`: caps maximum and minimum values at user defined values
-  :doc:`outliercappers/OutlierTrimmer`: removes outliers from the dataset

Scikit-learn Wrapper:
~~~~~~~~~~~~~~~~~~~~~

-  :doc:`wrappers/Wrapper`: executes Scikit-learn various transformers only on the selected subset of features

Mathematical Combination:
~~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`mathematical_combination/MathematicalCombinator`: applies basic mathematical operations across features

Feature Selection:
~~~~~~~~~~~~~~~~~~

- :doc:`selection/DropFeatures`: drops a subset of variables from a dataframe


Getting Help
------------

Can't get something to work? Here are places where you can find help.

1. The docs (you're here!).
2. `Stack Overflow <https://stackoverflow.com/questions/tagged/feature-engine>`_. If you ask a question, please tag it with "feature-engine".
3. If you are enrolled in the `Feature Engineering for Machine Learning course in Udemy <https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO>`_, post a question in a relevant section.
4. Join our `mailing list <https://groups.google.com/d/forum/feature-engine>`_.
5. Ask a question in the repo by filing an `issue <https://github.com/solegalli/feature_engine/issues/>`_.


Found a Bug or have a suggestion?
---------------------------------

Check if there's already an open `issue <https://github.com/solegalli/feature_engine/issues/>`_ on the topic. If not,
open a new `issue <https://github.com/solegalli/feature_engine/issues/>`_ with your bug report, suggestion or new feature request.


Open Source
-----------

Feature-engine's `license <https://github.com/solegalli/feature_engine/blob/master/LICENSE.md>`_ is an open source BSD 3-Clause.

Feature-engine is hosted on `GitHub <https://github.com/solegalli/feature_engine/>`_. The `issues <https://github.com/solegalli/feature_engine/issues/>`_ and `pull requests <https://github.com/solegalli/feature_engine/pulls>`_ are tracked there.


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents
   
   quickstart
   datasets
   imputers/index
   encoders/index
   vartransformers/index
   discretisers/index
   outliercappers/index
   wrappers/index
   mathematical_combination/index
   selection/index
   contributing/index
   code_of_conduct
   governance
   changelog