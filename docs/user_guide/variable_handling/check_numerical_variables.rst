.. _check_num_vars:

.. currentmodule:: feature_engine.variable_handling

check_numerical_variables
=========================

:class:`check_numerical_variables()` checks that the variables in the list are of
type numerical.

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

Let's now check that 2 of the variables are of type numerical:

.. code:: python

    from feature_engine.variable_handling import check_numerical_variables

    var_num = check_numerical_variables(df, ['Age', 'Marks'])

    var_num

If the variables are numerical, the function returns their names in a list:

.. code:: python

    ['Age', 'Marks']

If we pass a variable that is not of type numerical,
:class:`check_numerical_variables()` will return an error:

.. code:: python

    check_numerical_variables(df, ['Age', 'Name'])

Below we see the error message:

.. code:: python

    TypeError: Some of the variables are not numerical. Please cast them as numerical
    before using this transformer.
