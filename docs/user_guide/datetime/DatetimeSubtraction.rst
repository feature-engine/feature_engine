.. _datetime_subtraction:

.. currentmodule:: feature_engine.datetime

DatetimeSubtraction
===================

Very often, we have datetime variables in our datasets, and we want to determine the time
difference between them. For example, if we work with financial data, we may have the
variable **date of loan application**, with the date and time when the customer applied for
a loan, and also the variable **date of birth**, with the customer's date of birth. With those
two variables, we want to infer the **age of the customer** at the time of application. In order
to do this, we can compute the difference in years between `date_of_loan_application` and
`date_of_birth` and capture it in a new variable.

In a different example, if we are trying to predict the price of the house and we have
information about the year in which the house was built, we can infer the age of the house
at the point of sale. Generally, older houses cost less. To calculate the age of the house,
we’d simply compute the difference in years between the sale date and the date at which
it was built.

The Python program offers many options for making operations between datetime objects, like,
for example, the datetime module. Since most likely you will be working with Pandas dataframes,
we will focus this guide on pandas and then how we can automate the procedure with Feature-engine.

Subtracting datetime features with pandas
-----------------------------------------

In Python, we can subtract datetime objects with pandas. To work with datetime variables
in pandas, we need to make sure that the timestamp, which can be represented in various
formats, like strings (str), objects (`"O"`), or datetime, is cast as a datetime. If not, we
can convert strings to datetime objects by executing `pd.to_datetime(df[variable_of_interest])`.

Let’s create a toy dataframe with 2 datetime variables for a short demo:

.. code:: python

    import numpy as np
    import pandas as pd

    data = pd.DataFrame({
        "date1": pd.date_range("2019-03-05", periods=5, freq="D"),
        "date2": pd.date_range("2018-03-05", periods=5, freq="W")})

    print(data)

This is the data that we created, containing two datetime variables:

.. code:: python

           date1      date2
    0 2019-03-05 2018-03-11
    1 2019-03-06 2018-03-18
    2 2019-03-07 2018-03-25
    3 2019-03-08 2018-04-01
    4 2019-03-09 2018-04-08

Now, we can subtract `date2` from `date1` and capture the difference in a new variable by
utilizing the pandas subtraction operator:

.. code:: python

    data["diff"] = data["date1"].sub(data["date2"])

    print(data)

The new variable, which expresses the difference in number of days, is at the right of the
dataframe:

.. code:: python

           date1      date2     diff
    0 2019-03-05 2018-03-11 359 days
    1 2019-03-06 2018-03-18 353 days
    2 2019-03-07 2018-03-25 347 days
    3 2019-03-08 2018-04-01 341 days
    4 2019-03-09 2018-04-08 335 days

If we want the units in something other than days, we can use numpy’s timedelta. The following
example shows how to use this syntax:

.. code:: python

    data["diff"] = data["date1"].sub(data["date2"], axis=0).div(
        np.timedelta64(1, "Y").astype("timedelta64[ns]"))

    print(data)

We see the new variable now expressing the difference in years, at the right of the dataframe:

.. code:: python

           date1      date2      diff
    0 2019-03-05 2018-03-11  0.982909
    1 2019-03-06 2018-03-18  0.966481
    2 2019-03-07 2018-03-25  0.950054
    3 2019-03-08 2018-04-01  0.933626
    4 2019-03-09 2018-04-08  0.917199

If you wanted to subtract various datetime variables, you would have to write lines of code
for every subtraction. Fortunately, we can automate this procedure with :class:`DatetimeSubstraction()`.

Datetime subtraction with Feature-engine
----------------------------------------

:class:`DatetimeSubstraction()` automatically subtracts several date and time features from
each other. You just need to indicate the features at the right of the subtraction operation
in the `variables` parameters and those on the left in the `reference parameter`. You can also
change the output unit through the `output_unit` parameter.

:class:`DatetimeSubstraction()` works with variables whose `dtype` is datetime, as well as
with object-like and categorical variables, provided that they can be parsed into datetime
format. This will be done under the hood by the transformer.

Following up with the former example, here is how we obtain the difference in number of
days using :class:`DatetimeSubstraction()`:

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

With `transform()`, :class:`DatetimeSubstraction()` returns a new dataframe containing the
original variables and also the new variables with the time difference:

.. code:: python

           date1      date2  date1_sub_date2
    0 2019-03-05 2018-03-11         0.982909
    1 2019-03-06 2018-03-18         0.966481
    2 2019-03-07 2018-03-25         0.950054
    3 2019-03-08 2018-04-01         0.933626
    4 2019-03-09 2018-04-08         0.917199


Drop original variables after computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have the option to drop the original datetime variables after the computation:

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

In this case, the resulting dataframe contains only the time difference between the two
original variables:

.. code:: python

       date1_sub_date2
    0        11.794903
    1        11.597774
    2        11.400645
    3        11.203515
    4        11.006386

Subtract multiple variables simultaneously
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can perform multiple subtractions at the same time. In this example, we will add new
datetime variables to the toy dataframe as strings. The idea is to show that
:class:`DatetimeSubstraction()` will convert those strings to datetime under the hood to
carry out the subtraction operation.

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

The resulting dataframe contains the original variables plus the  new variables expressing
the time difference between the date objects.

.. code:: python

            date1       date2       date3       date4  date1_sub_date3  \
    0  2022-09-01  2022-09-15  2022-08-01  2022-08-15             31.0
    1  2022-10-01  2022-10-15  2022-09-01  2022-09-15             30.0
    2  2022-12-01  2022-12-15  2022-11-01  2022-11-15             30.0

       date2_sub_date3  date1_sub_date4  date2_sub_date4
    0             45.0             17.0             31.0
    1             44.0             16.0             30.0
    2             44.0             16.0             30.0


Working with missing values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, :class:`DatetimeSubstraction()`  will raise an error if the dataframe passed
to the `fit()` or `transform()` methods contains NA in the variables to subtract. We can
override this behaviour and allow computations between variables with nan by setting the
parameter `missing_values` to `"ignore"`. Here is a code example:

.. code:: python

    import numpy as np
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

When any of the variables contains NAN, the new features with the time difference will also
display NANs:

.. code:: python

            date1       date2       date3       date4  date1_sub_date3  \
    0  2022-09-01  2022-09-15  2022-08-01  2022-08-15             31.0
    1  2022-10-01         NaN  2022-09-01  2022-09-15             30.0
    2  2022-12-01  2022-12-15  2022-11-01         NaN             30.0

       date2_sub_date3  date1_sub_date4  date2_sub_date4
    0             45.0             17.0             31.0
    1              NaN             16.0              NaN
    2             44.0              NaN              NaN


Working with different timezones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we have timestamps in different timezones or variables in different timezones, we can
still perform subtraction operations with :class:`DatetimeSubstraction()` by first setting
all timestamps to the universal central time zone. Here is a code example, were we return
the time difference in microseconds:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeSubtraction

    data = pd.DataFrame({
        "date1": ['12:34:45+3', '23:01:02-6', '11:59:21-8', '08:44:23Z'],
        "date2": ['09:34:45+1', '23:01:02-6+1', '11:59:21-8-2', '08:44:23+3']
    })

    dfts = DatetimeSubtraction(
        variables="date1",
        reference="date2",
        utc=True,
        output_unit="ms",
        format="mixed"
    )

    new = dfts.fit_transform(data)

    print(new)

We see the resulting dataframe with the time difference in microseconds:

.. code:: python

            date1         date2  date1_sub_date2
    0  12:34:45+3    09:34:45+1        3600000.0
    1  23:01:02-6  23:01:02-6+1       25200000.0
    2  11:59:21-8  11:59:21-8-2       21600000.0
    3   08:44:23Z    08:44:23+3       10800000.0

Adding arbitrary names to the new variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, we want to compute just a few time differences. In this case, we may want as well
to assign the new variables specific names. In this code example, we do so:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeSubtraction

    data = pd.DataFrame({
        "date1": pd.date_range("2019-03-05", periods=5, freq="D"),
        "date2": pd.date_range("2018-03-05", periods=5, freq="W")})

    dtf = DatetimeSubtraction(
        variables="date1",
        reference="date2",
        new_variables_names=["my_new_var"]
        )

    data = dtf.fit_transform(data)

    print(data)

In the resulting dataframe, we see that the time difference was captured in a variable
called `my_new_var`:

.. code:: python

           date1      date2  my_new_var
    0 2019-03-05 2018-03-11       359.0
    1 2019-03-06 2018-03-18       353.0
    2 2019-03-07 2018-03-25       347.0
    3 2019-03-08 2018-04-01       341.0
    4 2019-03-09 2018-04-08       335.0

We should be mindful to pass a list of variales containing as many names as new variables.
The number of variables that will be created is obtained by multiplying the number of variables
in the parameter `variables` by the number of variables in the parameter `reference`.

get_feature_names_out()
~~~~~~~~~~~~~~~~~~~~~~~

Finally, we can extract the names of the transformed dataframe for compatibility with the
Scikit-learn pipeline:

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
    dtf.fit(data)

    dtf.get_feature_names_out()

Below the name of the variables that will appear in any dataframe resulting from applying
the `transform()` method:

.. code:: python

    ['date1',
     'date2',
     'date3',
     'date4',
     'date1_sub_date3',
     'date2_sub_date3',
     'date1_sub_date4',
     'date2_sub_date4']


Combining extraction and subtraction of datetime features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

In the following output we see the new dataframe contaning the features that were extracted
from the different datetime variables followed by those created by capturing the time
difference:

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


Additional resources
--------------------

For tutorials on how to create and use features from datetime columns, check the following courses:

.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

.. figure::  ../../images/fetsf.png
   :width: 300
   :figclass: align-center
   :align: right
   :target: https://www.trainindata.com/p/feature-engineering-for-forecasting

   Feature Engineering for Time Series Forecasting

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