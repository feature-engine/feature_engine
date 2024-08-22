.. _math_features:

.. currentmodule:: feature_engine.creation

MathFeatures
============

:class:`MathFeatures()` applies basic functions to groups of features, returning one or
more additional variables as a result.  It uses `pandas.agg()` to create the features,
so in essence, you can pass any function that is accepted by this method. One exception
is that :class:`MathFeatures()` does not accept dictionaries for the parameter `func`.

The functions can be passed as strings, numpy methods, i.e., np.mean, or any function
that you create, as long as, it returns a scalar from a vector.

For supported aggregation functions, see
`pandas documentation <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html>`_.

As an example, if we have the variables:

- **number_payments_first_quarter**
- **number_payments_second_quarter**
- **number_payments_third_quarter**
- **number_payments_fourth_quarter**

we can use :class:`MathFeatures()` to calculate the total number of payments
and mean number of payments as follows:

.. code-block:: python

    transformer = MathFeatures(
        variables=[
            'number_payments_first_quarter',
            'number_payments_second_quarter',
            'number_payments_third_quarter',
            'number_payments_fourth_quarter'
        ],
        func=['sum','mean'],
        new_variables_name=[
            'total_number_payments',
            'mean_number_payments'
        ]
    )

    Xt = transformer.fit_transform(X)


The transformed dataset, Xt, will contain the additional features
**total_number_payments** and **mean_number_payments**, plus the original set of
variables.

The variable **total_number_payments** is obtained by adding up the features
indicated in `variables`, whereas the variable **mean_number_payments** is
the mean of those 4 features.

Examples
--------

Let's dive into how we can use :class:`MathFeatures()` in more details. Let's first
create a toy dataset:

.. code:: python

    import numpy as np
    import pandas as pd
    from feature_engine.creation import MathFeatures

    df = pd.DataFrame.from_dict(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        })

    print(df)

The dataset looks like this:

.. code:: python

        Name        City  Age  Marks                 dob
    0    tom      London   20    0.9 2020-02-24 00:00:00
    1   nick  Manchester   21    0.8 2020-02-24 00:01:00
    2  krish   Liverpool   19    0.7 2020-02-24 00:02:00
    3   jack     Bristol   18    0.6 2020-02-24 00:03:00


We can now apply several functions over the numerical variables Age and Marks using
strings to indicate the functions:

.. code:: python

    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func = ["sum", "prod", "min", "max", "std"],
    )

    df_t = transformer.fit_transform(df)

    print(df_t)

And we obtain the following dataset, where the new variables are named after the function
used to obtain them, plus the group of variables that were used in the computation:

.. code:: python

        Name        City  Age  Marks                 dob  sum_Age_Marks  \
    0    tom      London   20    0.9 2020-02-24 00:00:00           20.9
    1   nick  Manchester   21    0.8 2020-02-24 00:01:00           21.8
    2  krish   Liverpool   19    0.7 2020-02-24 00:02:00           19.7
    3   jack     Bristol   18    0.6 2020-02-24 00:03:00           18.6

       prod_Age_Marks  min_Age_Marks  max_Age_Marks  std_Age_Marks
    0            18.0            0.9           20.0      13.505740
    1            16.8            0.8           21.0      14.283557
    2            13.3            0.7           19.0      12.940054
    3            10.8            0.6           18.0      12.303658


For more flexibility, we can pass existing functions to the `func` argument as follows:

.. code:: python

    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func = [np.sum, np.prod, np.min, np.max, np.std],
    )

    df_t = transformer.fit_transform(df)

    print(df_t)

And we obtain the following dataframe:

.. code:: python

        Name        City  Age  Marks                 dob  sum_Age_Marks  \
    0    tom      London   20    0.9 2020-02-24 00:00:00           20.9
    1   nick  Manchester   21    0.8 2020-02-24 00:01:00           21.8
    2  krish   Liverpool   19    0.7 2020-02-24 00:02:00           19.7
    3   jack     Bristol   18    0.6 2020-02-24 00:03:00           18.6

       prod_Age_Marks  amin_Age_Marks  amax_Age_Marks  std_Age_Marks
    0            18.0             0.9            20.0      13.505740
    1            16.8             0.8            21.0      14.283557
    2            13.3             0.7            19.0      12.940054
    3            10.8             0.6            18.0      12.303658

We have the option to set the parameter `drop_original` to True to drop the variables
after performing the calculations.

We can obtain the names of all the features in the transformed data as follows:

.. code:: python

    transformer.get_feature_names_out(input_features=None)

Which will return the names of all the variables in the transformed data:

.. code:: python

    ['Name',
     'City',
     'Age',
     'Marks',
     'dob',
     'sum_Age_Marks',
     'prod_Age_Marks',
     'amin_Age_Marks',
     'amax_Age_Marks',
     'std_Age_Marks']


New variables names
^^^^^^^^^^^^^^^^^^^

Even though the transfomer allows to combine variables automatically, its use is intended
to combine variables with domain knowledge. In this case, we normally want to
give meaningful names to the variables. We can do so through the parameter
`new_variables_names`.

`new_variables_names` takes a list of strings, with the new variable names. In this
parameter, you need to enter a list of names for the newly created features. You must
enter one name for each function indicated in the `func` parameter.
That is, if you want to perform mean and sum of features, you should enter 2 new
variable names. If you compute only the mean of features, enter 1 variable name.

The name of the variables should coincide with the order of the functions in `func`.
That is, if you set `func = ['mean', 'prod']`, the first new variable name will be
assigned to the mean of the variables and the second variable name to the product of the
variables.

Let's look at an example. In the following code snippet, we add up, and find the maximum
and minimum value of 2 variables, which results in 3 new features. We add the names
for the new features in a list:

.. code:: python

    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func = ["sum", "min", "max"],
        new_variables_names = ["sum_vars", "min_vars", "max_vars"]
    )

    df_t = transformer.fit_transform(df)

    print(df_t)

The resulting dataframe contains the new features under the variable names that we
provided:

.. code:: python

        Name        City  Age  Marks                 dob  sum_vars  min_vars  \
    0    tom      London   20    0.9 2020-02-24 00:00:00      20.9       0.9
    1   nick  Manchester   21    0.8 2020-02-24 00:01:00      21.8       0.8
    2  krish   Liverpool   19    0.7 2020-02-24 00:02:00      19.7       0.7
    3   jack     Bristol   18    0.6 2020-02-24 00:03:00      18.6       0.6

       max_vars
    0      20.0
    1      21.0
    2      19.0
    3      18.0


Additional resources
--------------------

For more details about this and other feature engineering methods check out these resources:


.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

|
|
|
|
|
|
|
|
|
|

Or read our book:

.. figure::  ../../images/cookbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587

   Python Feature Engineering Cookbook

|
|
|
|
|
|
|
|
|
|
|
|
|

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.