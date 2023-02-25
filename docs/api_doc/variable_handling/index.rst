.. -*- mode: rst -*-

Variable handling functions
===========================

Feature-engine functions for finding variables of a specific type or ensuring that the
variables are of the correct type.

The functions take a dataframe as an argument and return the names of the variables in
the desired type.

You can also pass a dataframe with a list of variables to check if they are all of the
desired type.

These functions are used under the hood by all Feature-engine transformers to select the
variables that they will modify or operate with.

.. toctree::
   :maxdepth: 1

   find_all_variables
   find_categorical_and_numerical_variables
   find_or_check_categorical_variables
   find_or_check_datetime_variables
   find_or_check_numerical_variables
