.. -*- mode: rst -*-

Changelog
=========

Version 0.6.0
-------------
Deployed: Friday, August 14, 2020

Contributors: 
    - Michał Gromiec
    - Surya Krishnamurthy
    - Gleb Levitskiy
    - Karthik Kothareddy
    - Richard Cornelius Suwandi
    - Chris Samiullah
    - Soledad Galli


Major Changes:
    - **New Transformer**: the ``MathematicalCombinator`` allows you combine multiple features into new variables by performing mathematical operations like sum, product, mean, standard deviation, or finding the minimum and maximum values (by Michał Gromiec).
    - **New Transformer**: the ``DropFeatures`` allows you remove specified variables from a dataset (by Karthik Kothareddy).
    - **New Transformer**: the ``DecisionTreeCategoricalEncoder`` encodes categorical variables with a decision tree (by Surya Krishnamurthy).
    - **Bug fix**: the ``SklearnTransformerWrapper`` can now automatically select numerical or numerical and categorical variables depending on the Scikit-learn transformer the user implements (by Michał Gromiec).
    - **Bug fix**: the ``SklearnTransformerWrapper`` can now wrap Scikit-learn's OneHotEncoder and concatenate the binary features back to the original dataframe (by Michał Gromiec).
    - **Added functionality**: the ``ArbitraryNumberImputer`` can now take a dictionary of variable, arbitrary number pairs, to impute different variables with different numbers (by Michał Gromiec).
    - **Added functionality**: the ``CategoricalVariableImputer`` can now replace missing data in categorical variables by a string defined by the user (by Gleb Levitskiy).
    - **Added functionality**: the ``RareLabelEnoder`` now allows the user to determine the maximum number of categories that the variable should have when grouping infrequent values (by Surya Krishnamurthy).


Minor Changes:
    - **Improved docs**: fixed typos and tidy Readme.md (by Richard Cornelius Suwandi)
    - **Improved engineering practices**: added Manifest.in to include md and licenses in tar ball in pypi (by Chris Samiullah)
    - **Improved engineering practices**: updated circleci yaml and created release branch for orchestrated release of new versions with significant changes (by Soledad Galli and Chris Samiullah)
    - **Improved engineering practices**: added test for doc build in circleci yaml (by Soledad Galli and Chris Samiullah)
    - **Transformer fix**: removed parameter return_object from the RareLabelEncoder as it was not working as intended(by Karthik Kothareddy and Soledad Galli)


Version 0.5.0
-------------

* Deployed: Friday, July 10, 2020
* Contributors: Soledad Galli

Major Changes:
    - **Bug fix**: fixed error in weight of evidence formula in the ``WoERatioCategoricalEncoder``. The old formula, that is np.log( p(1) / p(0) ) is preserved, and can be obtained by setting the ``encoding_method`` to 'log_ratio'. If ``encoding_method`` is set to 'woe', now the correct formula will operate.
	- **Added functionality**: most categorical encoders have the option ``inverse_transform``, to obtain the original value of the variable from the transformed dataset.
    - **Added functionality**: the `'Winsorizer``, ``OutlierTrimmer`` and ``ArbitraryOutlierCapper`` have now the option to ignore missing values, and obtain the parameters from the original variable distribution, or raise an error if the dataframe contains na, by setting the parameter ``missing_values`` to ``raise`` or ``ignore``.
    - **New Transformer**: the ``UserInputDiscretiser`` allows users to discretise numerical variables into arbitrarily defined buckets.


Version 0.4.3
-------------

* Deployed: Friday, May 15, 2020
* Contributors: Soledad Galli, Christopher Samiullah

Major Changes:
	- **New Transformer**: the `'SklearnTransformerWrapper`` allows you to use most Scikit-learn transformers just on a subset of features. Works with the SimpleImputer, the OrdinalEncoder and most scalers.

Minor Changes:
    - **Added functionality**: the `'EqualFrequencyDiscretiser`` and ``EqualWidthDiscretiser`` now have the ability to return interval boundaries as well as integers, to identify the bins. To return boundareis set the parameter ``return_boundaries=True``.
    - **Improved docs**: added contibuting section, where you can find information on how to participate in the development of Feature-engine's code base, and more.


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
