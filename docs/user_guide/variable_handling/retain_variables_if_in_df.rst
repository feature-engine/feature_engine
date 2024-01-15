.. _retain_vars:

.. currentmodule:: feature_engine.variable_handling

retain_variables_if_in_df
=========================

:class:`retain_variables_if_in_df()` returns the subset of variables in a list that
is present in the dataset.

Let's create a toy dataset with numerical, categorical and datetime variables:

.. code:: python

    import pandas as pd
    df = pd.DataFrame({
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
    })

    print(df.head())

We see the resulting dataframe below:

.. code:: python

        Name        City  Age  Marks                 dob
    0    tom      London   20    0.9 2020-02-24 00:00:00
    1   nick  Manchester   21    0.8 2020-02-24 00:01:00
    2  krish   Liverpool   19    0.7 2020-02-24 00:02:00
    3   jack     Bristol   18    0.6 2020-02-24 00:03:00

With :class:`retain_variables_if_in_df()` we capture in a list, the names of the
variables that are present in the dataset. So let's do that and then display the
resulting list:

.. code:: python

    from feature_engine.variable_handling import retain_variables_if_in_df

    vars_in_df = retain_variables_if_in_df(df, variables = ["Name", "City", "Dogs"])

    var_in_df

We see the names of the subset of variables that are in the dataframe below:

.. code:: python

    ['Name', 'City']

If none of variables in the list are in the dataset, :class:`retain_variables_if_in_df()`
will raise an error.

Uses
----

This function was originally developed for internal use.

When we run various feature selection transformers one after the other, for example,
`DropConstantFeatures`, then `DropDuplicateFeatures`, and finally
`RecursiveFeatureElimination`, we can't anticipate which variables will be dropped by
each transformer. Hence, these transformers use :class:`retain_variables_if_in_df()`
under the hood, to select those variables that were entered by the user and that still
remain in the dataset, before applying the selection algorithm.

We've now decided to expose this function as part of the `variable_handling` module. It
might be useful, for example, if you are creating `Feature-engine` compatible selection
transformers.
