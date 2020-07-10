.. feature_engine documentation master file, created by
   sphinx-quickstart on Wed Jan 10 14:43:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Feature-engine: A Feature Engineering for Machine Learning library
==================================================================
.. figure::  images/FeatureEngine.png
   :align:   center

   Feature-engine rocks!

Feature-engine is a Python library with multiple transformers to engineer features for use
in machine learning models. Feature-engine preserves Scikit-learn functionality with fit() and 
transform() methods to learn parameters from and then transform data.

Feature-engine includes transformers for:

- Missing value imputation
- Categorical variable encoding
- Outlier capping
- Discretisation
- Numerical variable transformation

Feature-engine allows you to select the variables to engineer within each transformer. This way,
different engineering procedures can be easily applied to different feature subsets.

Feature-engine's transformers can be assembled within the Scikit-learn pipeline, therefore making it
possible to save and deploy one single object (.pkl) with the entire machine learning pipeline.

More details into what is unique about Feature-engine can be found in this article:
`Feature-engine: A new open source Python package for feature engineering <<https://www.trainindata.com/post/feature-engine-a-new-open-source-python-package-for-feature-engineering>`_.


Installation
------------

Feature-engine is a Python 3 package and works well with 3.6 or later. Earlier versions have not been tested.
The simplest way to install Feature-engine is from PyPI with pip, Python's preferred package installer.

.. code-block:: bash

    $ pip install feature-engine


Contributing
------------

Interested in contributing to Feature-engine? That is great news! Feature-engine is a welcoming and inclusive
project and it would be great to have you onboard. We follow the `Python Software Foundation Code of Conduct <http://www.python.org/psf/codeofconduct/>`_.

Regardless of your skill level you can help us. We appreciate bug reports, user testing, feature requests, bug fixes, addition of tests, product enhancements, and documentation improvements.

For more details on how to contribute check the contributing page. Click on the "Contributing" link in the
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


Outlier Capping: Cappers
~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`outliercappers/Winsorizer`: caps maximum or minimum values using statistical parameters
-  :doc:`outliercappers/ArbitraryOutlierCapper`: caps maximum and minimum values at user defined values
-  :doc:`outliercappers/OutlierTrimmer`: removes outliers from the dataset

Scikit-learn Wrapper:
~~~~~~~~~~~~~~~~~~~~~

-  :doc:`wrappers/Wrapper`: executes Scikit-learn various transformers only on the selected subset of features

Getting Help
------------

Can't get something to work? Here are places you can find help.

1. The docs (you're here!).
2. `Stack Overflow <https://stackoverflow.com/questions/tagged/feature-engine>`_. If you ask a question, please tag it with "feature-engine".
3. If you are enrolled in the `Feature Engineering for Machine Learning course in Udemy <https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=FEATENGREPO>`_, post a question in a relevant section.

Find a Bug?
-----------

Check if there's already an open `issue <https://github.com/solegalli/feature_engine/issues/>`_ on the topic. If needed, file an `issue <https://github.com/solegalli/feature_engine/issues/>`_.


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
   contributing/index
   changelog