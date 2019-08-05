.. feature_engine documentation master file, created by
   sphinx-quickstart on Wed Jan 10 14:43:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Feature-engine: A Feature Engineering for Machine Learning library
==================================================================
.. figure::  images/FeatureEngine.png
   :align:   center

   Feature-engine rocks!

Feature-engine is a Python library that contains several transformers to engineer features for use
in machine learning models. Feature-engine preserves Scikit-learn functionality with fit() and 
transform() methods to learn parameters from and then transform data.

Feature-engine includes transformers for:

- Missing value imputation
- Categorical variable encoding
- Outlier capping
- Discretisation
- Numerical variable transformation

Feature-engine allows to select which variables to engineer within each transformer.

Feature-engine's transformers can be assembled within the Scikit-learn pipeline, therefore making it
possible to save and deploy one single object (.pkl) with the entire machine learning pipeline.


Installation
------------

Feature-engine is a Python 3 package and works well with 3.5 or later. Earlier versions have not been tested.
The simplest way to install Feature-engine is from PyPI with pip, Python's preferred package installer.

.. code-block:: bash

    $ pip install feature-engine


Contributing
------------

Interested in contributing to Feature-engine? That is great news! Feature-engine is a welcoming and inclusive
project and it would be great to have you onboard. We follow the `Python Software Foundation Code of Conduct <http://www.python.org/psf/codeofconduct/>`_.

Regardless of your skill level you can help us. We appreciate bug reports, user testing, feature requests, bug fixes, addition of tests, product enhancements, and documentation improvements.

More details on how to contribute will come soon! Meanwhile, feel free to fork the Github repo and make pull requests,
create an issue, or send feedback. More details on how to reach us in the **Getting help** section below.

Thank you for your contributions!

Feature-engine's Transformers
-----------------------------
Missing Data Imputation: Imputers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`imputers/MeanMedianImputer`: replaces missing data in numerical variables by mean or median 
- :doc:`imputers/ArbitraryValueImputer`: replaces missing data in numerical variables by an arbitrary value
- :doc:`imputers/EndTailImputer`: replaces missing data in numerical variables by numbers at the distribution tails
- :doc:`imputers/CategoricalVariableImputer`: replaces missing data in categorical variables with the string 'Missing'
- :doc:`imputers/FrequentCategoryImputer`: replaces missing data in categorical variables by the mode
- :doc:`imputers/RandomSampleImputer`: replaces missing data with random samples of the variable
- :doc:`imputers/AddNaNBinaryImputer`: adds a binary missing indicator to flag observations with missing data

Categorical Variable Encoders: Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`encoders/OneHotCategoricalEncoder`: performs one hot encoding, optional: of popular categories
- :doc:`encoders/CountFrequencyCategoricalEncoder`: replaces categories by observation number or percentage
- :doc:`encoders/OrdinalCategoricalEncoder`: replaces categories by numbers arbitrarily or ordered by target
- :doc:`encoders/MeanCategoricalEncoder`: replaces categories by the target mean
- :doc:`encoders/WoERatioCategoricalEncoder`: replaces categories by the weight of evidence
- :doc:`encoders/RareLabelCategoricalEncoder`: groups infrequent categories in one group

Numerical Variable Transformation: Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`vartransformers/LogTransformer`: perform logarithmic transformation of numerical variables
- :doc:`vartransformers/ReciprocalTransformer`: perform reciprocal transformation of numerical variables
- :doc:`vartransformers/PowerTransformer`: perform power transformation of numerical variables
- :doc:`vartransformers/BoxCoxTransformer`: performs Box-Cox transformation of numerical variables
- :doc:`vartransformers/YeoJohnsonTransformer`: performs Yeo-Johnson transformation of numerical variables

Variable Discretisation: Discretisers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`discretisers/EqualFrequencyDiscretiser`: sorts variable into equal percentage of obs intervals
- :doc:`discretisers/EqualWidthDiscretiser`: sorts variable into equal size contiguous intervals
- :doc:`discretisers/DecisionTreeDiscretiser`: uses decision trees to create finite variables

Outlier Capping: Cappers
~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`outliercappers/Winsorizer`: caps maximum or minimum values using Gaussian approx or IQR rule
-  :doc:`outliercappers/ArbitraryOutlierCapper`: caps maximum and minimum values arbitrarily


Getting Help
------------

Can't get someting to work? Here are places you can find help.

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
   imputers/index
   encoders/index
   vartransformers/index
   discretisers/index
   outliercappers/index
   changelog