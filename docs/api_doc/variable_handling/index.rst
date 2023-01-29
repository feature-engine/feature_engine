.. -*- mode: rst -*-

Variable handling functions
===========================

Feature-engine functions for finding variables of a specific type or ensuring that the
variables are of the correct type.

The functions take a dataframe as an argument and return the names of the variables in
the desired type.

You can also pass a dataframe with a list of variables to check if they are all of the
desired type.

.. currentmodule:: feature_engine.variable_handling.variable_type_selection

.. autosummary::
   :toctree: generated/

    find_all_variables
    find_categorical_and_numerical_variables
    find_or_check_categorical_variables
    find_or_check_datetime_variables
    find_or_check_numerical_variables
