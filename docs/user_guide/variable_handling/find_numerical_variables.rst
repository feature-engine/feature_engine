.. _find_num_vars:

.. currentmodule:: feature_engine.variable_handling

find_numerical_variables
========================

:class:`find_numerical_variables()` returns a list with the names of the numerical
variables in the dataset.

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

With :class:`find_numerical_variables()` we capture the names of all numerical
variables in a list. So let's do that and then display the list:

.. code:: python

    from feature_engine.variable_handling import find_numerical_variables

    var_num = find_numerical_variables(df)

    var_num

We see the names of the numerical variables in the list below:

.. code:: python

    ['Age', 'Marks']

If there are no numerical variables in the dataset, :class:`find_numerical_variables()`
will raise an error.
