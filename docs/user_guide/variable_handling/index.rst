.. -*- mode: rst -*-

Variable handling functions
===========================

This set of functions find variables of a specific type in a dataframe, or check that a
list of variables is of a specified data type.

The `find` functions take a dataframe as an argument and returns a list with the names
of the variables of the desired type.

The `check` functions check that the list of variables are all of the desired data type.

The `retain` functions select the variables in a list if they fulfill a condition.

You can use these functions to identify different sets of variables based on their
data type to streamline your feature engineering pipelines or create your own
Feature-engine or Scikit-learn compatible transformers.


.. toctree::
   :maxdepth: 1

   find_all_variables
   find_categorical_variables
   find_datetime_variables
   find_numerical_variables
   find_categorical_and_numerical_variables
   check_all_variables
   check_categorical_variables
   check_datetime_variables
   check_numerical_variables
   retain_variables_if_in_df
