.. _relative_features:

.. currentmodule:: feature_engine.creation

RelativeFeatures
================

:class:`RelativeFeatures()` applies basic mathematical operations between a group
of variables and one or more reference features, adding the resulting features
to the dataframe.

:class:`RelativeFeatures()` uses the pandas methods `pd.DataFrame.add()`, `pd.DataFrame.sub()`,
`pd.DataFrame.mul()`, `pd.DataFrame.div()`, `pd.DataFrame.truediv()`, `pd.DataFrame.floordiv()`,
`pd.DataFrame.mod()` and `pd.DataFrame.pow()` to transform a group of variables by a group
of reference variables.

For example, if we have the variables:

- **number_payments_first_quarter**
- **number_payments_second_quarter**
- **number_payments_third_quarter**
- **number_payments_fourth_quarter**
- **total_payments**,

we can use :class:`RelativeFeatures()` to determine the percentage of payments per
quarter as follows:

.. code-block:: python

    transformer = RelativeFeatures(
        variables=[
            'number_payments_first_quarter',
            'number_payments_second_quarter',
            'number_payments_third_quarter',
            'number_payments_fourth_quarter',
        ],
        reference=['total_payments'],
        func=['div'],
    )

    Xt = transformer.fit_transform(X)

The precedent code block will return a new dataframe, Xt, with 4 new variables that are
calculated as the division of each one of the variables in `variables` and
'total_payments'.

Examples
--------

Let's dive into how we can use :class:`RelativeFeatures()` in more details. Let's first
create a toy dataset:

.. code:: python

    import pandas as pd
    from feature_engine.creation import RelativeFeatures

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

We can now apply several functions between the numerical variables Age and Marks and Age
as follows:

.. code:: python

    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age"],
        func = ["sub", "div", "mod", "pow"],
    )

    df_t = transformer.fit_transform(df)

    print(df_t)

And we obtain the following dataset, where the new variables are named after the variables
that were used for the calculation and the function in the middle of their names. Thus,
`Mark_sub_Age` means `Mark - Age`, and `Marks_mod_Age` means `Mark % Age`.

.. code:: python

        Name        City  Age  Marks                 dob  Age_sub_Age  \
    0    tom      London   20    0.9 2020-02-24 00:00:00            0
    1   nick  Manchester   21    0.8 2020-02-24 00:01:00            0
    2  krish   Liverpool   19    0.7 2020-02-24 00:02:00            0
    3   jack     Bristol   18    0.6 2020-02-24 00:03:00            0

       Marks_sub_Age  Age_div_Age  Marks_div_Age  Age_mod_Age  Marks_mod_Age  \
    0          -19.1          1.0       0.045000            0            0.9
    1          -20.2          1.0       0.038095            0            0.8
    2          -18.3          1.0       0.036842            0            0.7
    3          -17.4          1.0       0.033333            0            0.6

               Age_pow_Age  Marks_pow_Age
    0 -2101438300051996672       0.121577
    1 -1595931050845505211       0.009223
    2  6353754964178307979       0.001140
    3  -497033925936021504       0.000102


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
     'Age_sub_Age',
     'Marks_sub_Age',
     'Age_div_Age',
     'Marks_div_Age',
     'Age_mod_Age',
     'Marks_mod_Age',
     'Age_pow_Age',
     'Marks_pow_Age']


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