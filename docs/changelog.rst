.. -*- mode: rst -*-

Changelog
=========

Version 0.4.3
-------------
* Deployed: , 2020
* Contributors: Soledad Galli

Major Changes:


Minor Changes:
    - **Added functionality**: the `'EqualFrequencyDiscretiser`` and ``EqualWidthDiscretiser`` now have the ability to return interval boundaries as well as integers, to identify the bins. To return boundareis set the parameter ``return_boundaries=True``.

Version 0.4.0
-------------
* Deployed: Monday, April 04, 2020
* Contributors: Soledad Galli, Christopher Samiullah

Major Changes:
    - **Deprecated**: the ``FrequentCategoryImputer`` was integrated into the class ``CategoricalVariableImputer``. To perform frequent category imputation now use: ``CategoricalVariableImputer(imputation_method='frequent')``
    - **Renamed**: the ``AddNaNBinaryImputer`` is now called ``AddMissingIndicator``.
    - **New**: the ``OutlierTrimmer`` was introduced into the package and allows you to remove outliers from the dataset

Minor Changes:
    - **Improved**: the ``EndTailImputer`` now has the additional option to place outliers at a factor of the maximum value.
    - **Improved**: the ``FrequentCategoryImputer`` has now the functionality to return numerical variables cast as object, in case you want to operate with them as if they were categorical. Set ``return_object=True``.
    - **Improved**: the ``RareLabelEncoder`` now allows the user to define the name for the label that will replace rare categories.
    - **Improved**: All feature engine transformers (except missing data imputers) check that the data sets do not contain missing values.
    - **Improved**: the ``LogTransformer`` will raise an error if a variable has zero or negative values.
    - **Improved**: the ``ReciprocalTransformer`` now works with variables of type integer.
    - **Improved**: the ``ReciprocalTransformer`` will raise an error if the variable contains the value zero.
    - **Improved**: the ``BoxCoxTransformer`` will raise an error if the variable contains negative values.
    - **Improved**: the ``OutlierCapper`` now finds and removes outliers based of percentiles.
    - **Improved**: Feature-engine is now compatible with latest releases of Pandas and Scikit-learn.


Version 0.3.0
-------------
* Deployed: Monday, August 05, 2019
* Contributors: Soledad Galli.

Major Changes:
    - **New**: the ``RandomSampleImputer`` now has the option to set one seed for batch imputation or set a seed observation per observations based on 1 or more additional numerical variables for that observation, which can be combined with multiplication or addition.
    - **New**: the ``YeoJohnsonTransfomer`` has been included to perform Yeo-Johnson transformation of numerical variables.
    - **Renamed**: the  ``ExponentialTransformer`` is now called ``PowerTransformer``.
    - **Improved**: the ``DecisionTreeDiscretiser`` now allows to provide a grid of parameters to tune the decision trees which is done with a GridSearchCV under the hood.
    - **New**: Extended documentation for all Feature-engine's transformers.
    - **New**:  *Quickstart* guide to jump on straight onto how to use Feature-engine.
    - **New**: *Changelog* to track what is new in Feature-engine.
    - **Updated**: new ``Jupyter notebooks`` with examples on how to use Feature-engine's transformers.

Minor Changes:
    - **Unified**: dictionary attributes in transformers, which contain the transformation mappings, now end with ``_``, for example ``binner_dict_``.