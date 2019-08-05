.. -*- mode: rst -*-

Changelog
=========

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