.. _datetime_subtraction:

.. currentmodule:: feature_engine.datetime

DatetimeSubtraction
===================

Very often we have datetime variables in our datasets and we want to determine the
time elapsed between them. For example, if we work with financial data, we may have the
variable `date_of_loan_application` with the date and time when the customer applied
for a loan, and also the variable `date_of_birth`, with the customers' date of birth.
With those 2 variables, we want to infer the **age** of the customer at the time of application.
In order to do this, we can compute the difference in years between `date_of_loan_application`
and `date_of_birth` and capture it in a new variable.

In a different example, if we are trying to predict the price of the house and we have
information about the year in which the house was built, we can infer the age of the house
at the point of sale. Generally, older houses cost less.

Subtracting datetime features with pandas
-----------------------------------------

In Python, we can subtract datetime variables with pandas. Let's create a toy dataframe
with 2 datetime variables first:

.. code:: python

    import numpy as np
    import pandas as pd

    data = pd.DataFrame({
        "date1": pd.date_range("2019-03-05", periods=5, freq="D"),
        "date2": pd.date_range("2018-03-05", periods=5, freq="W")})

    print(data)

This is the data that we created:

.. code:: python

           date1      date2
    0 2019-03-05 2018-03-11
    1 2019-03-06 2018-03-18
    2 2019-03-07 2018-03-25
    3 2019-03-08 2018-04-01
    4 2019-03-09 2018-04-08

Now, let's subtract `date2` from `date1` and capture the difference in a new variable:

.. code:: python

    data["diff"] = data["date1"].sub(data["date2"])

    print(data)

We see the new variable at the right of the dataframe:

.. code:: python

           date1      date2     diff
    0 2019-03-05 2018-03-11 359 days
    1 2019-03-06 2018-03-18 353 days
    2 2019-03-07 2018-03-25 347 days
    3 2019-03-08 2018-04-01 341 days
    4 2019-03-09 2018-04-08 335 days

If we want the units in something different than days, we can use `numpy`'s timedelta:

.. code:: python

data["diff"] = data["date1"].sub(data["date2"], axis=0).apply(
    lambda x: x / np.timedelta64(1, "Y"))

print(data)

We see the new variable now expressing the difference in years, at the right of the dataframe:

.. code:: python

           date1      date2      diff
    0 2019-03-05 2018-03-11  0.982909
    1 2019-03-06 2018-03-18  0.966481
    2 2019-03-07 2018-03-25  0.950054
    3 2019-03-08 2018-04-01  0.933626
    4 2019-03-09 2018-04-08  0.917199

We can automate this procedure with ::class:`DatetimeSubstraction()`.

Datetime subtraction with Feature-engine
----------------------------------------

:class:`DatetimeFeatures()` automatically subtracts several date and time features from
each other. You just need to indicate the features at the right of the subtraction operation
in the `variables` parameters, and those on the left in the `reference` parameter. You can
also change the output unit through the `output_unit` parameter.

It works with variables whose dtype is datetime, as well as with object-like and categorical
variables, provided that they can be parsed into datetime format.

Following up with the former example:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeSubtraction

    data = pd.DataFrame({
        "date1": pd.date_range("2019-03-05", periods=5, freq="D"),
        "date2": pd.date_range("2018-03-05", periods=5, freq="W")})

    dtf = DatetimeSubtraction(
        variables="date1",
        reference="date2",
        output_unit="Y")

    data = dtf.fit_transform(data)

    print(data)

We see the new variable expressing the difference in years at the right of the dataframe:

.. code:: python

           date1      date2  date1_sub_date2
    0 2019-03-05 2018-03-11         0.982909
    1 2019-03-06 2018-03-18         0.966481
    2 2019-03-07 2018-03-25         0.950054
    3 2019-03-08 2018-04-01         0.933626
    4 2019-03-09 2018-04-08         0.917199


We can also drop the original datetime variables after the computation:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeSubtraction

    data = pd.DataFrame({
        "date1": pd.date_range("2019-03-05", periods=5, freq="D"),
        "date2": pd.date_range("2018-03-05", periods=5, freq="W")})

    dtf = DatetimeSubtraction(
        variables="date1",
        reference="date2",
        output_unit="M",
        drop_original=True
    )

    data = dtf.fit_transform(data)

    print(data)

.. code:: python

       date1_sub_date2
    0        11.794903
    1        11.597774
    2        11.400645
    3        11.203515
    4        11.006386


We can perform multiple subtractions at the same time:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeSubtraction

    data = pd.DataFrame({
        "date1" : ["2022-09-01", "2022-10-01", "2022-12-01"],
        "date2" : ["2022-09-15", "2022-10-15", "2022-12-15"],
        "date3" : ["2022-08-01", "2022-09-01", "2022-11-01"],
        "date4" : ["2022-08-15", "2022-09-15", "2022-11-15"],
    })

    dtf = DatetimeSubtraction(variables=["date1", "date2"], reference=["date3", "date4"])

    data = dtf.fit_transform(data)

    print(data)

.. code:: python

            date1       date2       date3       date4  date1_sub_date3  \
    0  2022-09-01  2022-09-15  2022-08-01  2022-08-15             31.0
    1  2022-10-01  2022-10-15  2022-09-01  2022-09-15             30.0
    2  2022-12-01  2022-12-15  2022-11-01  2022-11-15             30.0

       date2_sub_date3  date1_sub_date4  date2_sub_date4
    0             45.0             17.0             31.0
    1             44.0             16.0             30.0
    2             44.0             16.0             30.0


We can work with variables with nan:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeSubtraction

    data = pd.DataFrame({
        "date1" : ["2022-09-01", "2022-10-01", "2022-12-01"],
        "date2" : ["2022-09-15", np.nan, "2022-12-15"],
        "date3" : ["2022-08-01", "2022-09-01", "2022-11-01"],
        "date4" : ["2022-08-15", "2022-09-15", np.nan],
    })

    dtf = DatetimeSubtraction(
        variables=["date1", "date2"],
        reference=["date3", "date4"],
        missing_values="ignore")

    data = dtf.fit_transform(data)

    print(data)



.. code:: python

            date1       date2       date3       date4  date1_sub_date3  \
    0  2022-09-01  2022-09-15  2022-08-01  2022-08-15             31.0
    1  2022-10-01         NaN  2022-09-01  2022-09-15             30.0
    2  2022-12-01  2022-12-15  2022-11-01         NaN             30.0

       date2_sub_date3  date1_sub_date4  date2_sub_date4
    0             45.0             17.0             31.0
    1              NaN             16.0              NaN
    2             44.0              NaN              NaN


Finally, we can extract the names of the transformed dataframe for compatibility with the
Scikit-learn pipeline:

.. code:: python

    dtf.get_feature_names_out()

.. code:: python

    ['date1',
     'date2',
     'date3',
     'date4',
     'date1_sub_date3',
     'date2_sub_date3',
     'date1_sub_date4',
     'date2_sub_date4']


We can also combine the creation of numerical variables from datetime features with the
creation of new features by subtraction of datetime variables:

.. code:: python

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from feature_engine.datetime import DatetimeFeatures, DatetimeSubtraction

    data = pd.DataFrame({
        "date1" : ["2022-09-01", "2022-10-01", "2022-12-01"],
        "date2" : ["2022-09-15", "2022-10-15", "2022-12-15"],
        "date3" : ["2022-08-01", "2022-09-01", "2022-11-01"],
        "date4" : ["2022-08-15", "2022-09-15", "2022-11-15"],
    })

    dtf = DatetimeFeatures(variables=["date1", "date2"], drop_original=False)
    dts = DatetimeSubtraction(
        variables=["date1", "date2"],
        reference=["date3", "date4"],
        drop_original=True,
    )

    pipe = Pipeline([
        ("features", dtf),("subtraction", dts)
    ])

    data = pipe.fit_transform(data)

    print(data)


.. code:: python

       date1_month  date1_year  date1_day_of_week  date1_day_of_month  date1_hour  \
    0            9        2022                  3                   1           0
    1           10        2022                  5                   1           0
    2           12        2022                  3                   1           0

       date1_minute  date1_second  date2_month  date2_year  date2_day_of_week  \
    0             0             0            9        2022                  3
    1             0             0           10        2022                  5
    2             0             0           12        2022                  3

       date2_day_of_month  date2_hour  date2_minute  date2_second  \
    0                  15           0             0             0
    1                  15           0             0             0
    2                  15           0             0             0

       date1_sub_date3  date2_sub_date3  date1_sub_date4  date2_sub_date4
    0             31.0             45.0             17.0             31.0
    1             30.0             44.0             16.0             30.0
    2             30.0             44.0             16.0             30.0

For tutorials on how to create and use features from datetime columns, check the following courses:

- `Feature Engineering for Machine Learning <https://www.trainindata.com/p/feature-engineering-for-machine-learning>`_.
- `Feature Engineering for Time Series Forecasting <https://www.courses.trainindata.com/p/feature-engineering-for-forecasting>`_.