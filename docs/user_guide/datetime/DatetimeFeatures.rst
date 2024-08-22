.. _datetime_features:

.. currentmodule:: feature_engine.datetime

DatetimeFeatures
================

In datasets commonly used in data science and machine learning projects, the variables very
often contain information about date and time. **Date of birth** and **time of purchase** are two
examples of these variables. They are commonly referred to as “datetime features”, that is,
data whose data type is date and time.

We don’t normally use datetime variables in their raw format to train machine learning models,
like those for regression, classification, or clustering. Instead, we can extract a lot of information
from these variables by extracting the different date and time components of the datetime
variable.

Examples of date and time components are the year, the month, the week_of_year, the day
of the week, the hour, the minutes, and the seconds.

Datetime features with pandas
-----------------------------

In Python, we can extract date and time components through the `dt` module of the open-source
library pandas. For example, by executing the following:

.. code:: python

    data = pd.DataFrame({"date": pd.date_range("2019-03-05", periods=20, freq="D")})

    data["year"] = data["date"].dt.year
    data["quarter"] = data["date"].dt.quarter
    data["month"] = data["date"].dt.month

In the former code block we created 3 features from the timestamp variable: the *year*, the
*quarter* and the *month*.


Datetime features with Feature-engine
-------------------------------------

:class:`DatetimeFeatures()` automatically extracts several date and time features from
datetime variables. It works with variables whose dtype is datetime, as well as with
object-like and categorical variables, provided that they can be parsed into datetime
format. It *cannot* extract features from numerical variables.

:class:`DatetimeFeatures()` uses the pandas `dt` module under the hood, therefore automating
datetime feature engineering. In two lines of code and by specifying which features we
want to create with :class:`DatetimeFeatures()`, we can create multiple date and time variables
from various variables simultaneously.

:class:`DatetimeFeatures()` can automatically create all features supported by pandas `dt`
and a few more, like, for example, a binary feature indicating if the event occurred on
a weekend and also the semester.

With :class:`DatetimeFeatures()` we can choose which date and time features to extract
from the datetime variables. We can also extract date and time features from one or more
datetime variables at the same time.

Through the following examples we highlight the functionality and versatility of :class:`DatetimeFeatures()`
for tabular data.

Extract date features
~~~~~~~~~~~~~~~~~~~~~

In this example, we are going to extract three **date features** from a
specific variable in the dataframe. In particular, we are interested
in the *month*, the *day of the year*, and whether that day was the *last
day the month*.

First, we will create a toy dataframe with 2 date variables:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_date1": ['May-1989', 'Dec-2020', 'Jan-1999', 'Feb-2002'],
        "var_date2": ['06/21/2012', '02/10/1998', '08/03/2010', '10/31/2020'],
    })

Now, we will extract the variables month, month-end and the day of the year from the
second datetime variable in our dataset.

.. code:: python

    dtfs = DatetimeFeatures(
        variables="var_date2",
        features_to_extract=["month", "month_end", "day_of_year"]
    )

    df_transf = dtfs.fit_transform(toy_df)

    df_transf

With `transform()`, the features extracted from the datetime variable are added to the
dataframe.

We see the new features in the following output:

.. code:: python

      var_date1  var_date2_month  var_date2_month_end  var_date2_day_of_year
    0  May-1989                6                    0                    173
    1  Dec-2020                2                    0                     41
    2  Jan-1999                8                    0                    215
    3  Feb-2002               10                    1                    305

By default, :class:`DatetimeFeatures()` drops the variable from which the date and time
features were extracted, in this case, *var_date2*. To keep the variable, we just need
to indicate `drop_original=False` when initializing the transformer.

Finally, we can obtain the name of the variables in the returned data as follows:

.. code:: python

    dtfs.get_feature_names_out()

.. code:: python

    ['var_date1',
     'var_date2_month',
     'var_date2_month_end',
     'var_date2_day_of_year']


Extract time features
~~~~~~~~~~~~~~~~~~~~~

In this example, we are going to extract the feature *minute* from the two time
variables in our dataset.

First, let's create a toy dataset with 2 time variables and an object variable.

.. code:: python 

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "not_a_dt": ['not', 'a', 'date', 'time'],
        "var_time1": ['12:34:45', '23:01:02', '11:59:21', '08:44:23'],
        "var_time2": ['02:27:26', '10:10:55', '17:30:00', '18:11:18'],
    })

:class:`DatetimeFeatures()` automatically finds all variables that can be parsed to
datetime. So if we want to extract time features from all our datetime variables, we
don't need to specify them.

Note that from version 2.0.0 pandas deprecated the parameter `infer_datetime_format`.
Hence, if you want pandas to infer the datetime format and you have different formats,
you need to explicitly say so by passing `"mixed"` to the `format` parameter as shown
below.

.. code:: python

    dfts = DatetimeFeatures(features_to_extract=["minute"], format="mixed")

    df_transf = dfts.fit_transform(toy_df)

    df_transf

We see the new features in the following output:

.. code:: python

      not_a_dt  var_time1_minute  var_time2_minute
    0      not                34                27
    1        a                 1                10
    2     date                59                30
    3     time                44                11


The transformer found two variables in the dataframe that can be cast to datetime and
proceeded to extract the requested feature from them.

The variables detected as datetime are stored in the transformer's `variables_` attribute:

.. code:: python

    dfts.variables_

.. code:: python

    ['var_time1', 'var_time2']

The original datetime variables are dropped from the data by default. This leaves the
dataset ready to train machine learning algorithms like linear regression or random forests.

If we want to keep the datetime variables, we just need to indicate `drop_original=False`
when initializing the transformer.

Finally, if we want to obtain the names of the variables in the output data, we can use:

.. code:: python

    dfts.get_feature_names_out()

.. code:: python

    ['not_a_dt', 'var_time1_minute', 'var_time2_minute']


Extract date and time features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will combine what we have seen in the previous two examples
and extract a date feature - *year* - and time feature - *hour* -
from two variables that contain both date and time information.

Let's go ahead and create a toy dataset with 3 datetime variables.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_dt1": pd.date_range("2018-01-01", periods=3, freq="H"),
        "var_dt2": ['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21'],
        "var_dt3": ['03/02/15 02:27:26', '02/28/97 10:10:55', '11/11/03 17:30:00'],
    })

Now, we set up the :class:`DatetimeFeatures()` to extract features from 2 of the datetime
variables. In this case, we do not want to drop the datetime variable after extracting
the features.

.. code:: python

    dfts = DatetimeFeatures(
        variables=["var_dt1", "var_dt3"],
        features_to_extract=["year", "hour"],
        drop_original=False,
        format="mixed",
    )
    df_transf = dfts.fit_transform(toy_df)

    df_transf

We can see the resulting dataframe in the following output:

.. code:: python

                  var_dt1            var_dt2            var_dt3  var_dt1_year  \
    0 2018-01-01 00:00:00  08/31/00 12:34:45  03/02/15 02:27:26          2018
    1 2018-01-01 01:00:00  12/01/90 23:01:02  02/28/97 10:10:55          2018
    2 2018-01-01 02:00:00  04/25/01 11:59:21  11/11/03 17:30:00          2018

       var_dt1_hour  var_dt3_year  var_dt3_hour
    0             0          2015             2
    1             1          1997            10
    2             2          2003            17

And that is it. The new features are now added to the dataframe.

Time series
~~~~~~~~~~~

Time series data consists of datapoints indexed in time order. The time is usually in
the index of the dataframe. We can extract features from the timestamp index and use them
for time series regression or classification, as well as for time series forecasting.

With :class:`DatetimeFeatures()` we can also create date and time features from the
dataframe index.

Let's create a toy dataframe with datetime in the index.

.. code:: python

    import pandas as pd

    X = {"ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68],
         "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52],
         "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56],
         "color": ["green"] * 4 + ["blue"] * 4,
         }

    X = pd.DataFrame(X)
    X.index = pd.date_range("2020-05-15 12:00:00", periods=8, freq="15min")

    X.head()

Below we see the output of our toy dataframe:

.. code:: python

                         ambient_temp  module_temp  irradiation  color
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

We can extract features from the index as follows:

..  code:: python

    from feature_engine.datetime import DatetimeFeatures

    dtf = DatetimeFeatures(variables="index")

    Xtr = dtf.fit_transform(X)

    Xtr

We can see that the transformer created the default time features and added them at
the end of the dataframe.

.. code:: python

                         ambient_temp  module_temp  irradiation  color  month  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green      5
    2020-05-15 12:15:00         31.51        49.84         0.79  green      5
    2020-05-15 12:30:00         32.15        52.35         0.65  green      5
    2020-05-15 12:45:00         32.39        50.63         0.76  green      5
    2020-05-15 13:00:00         32.62        49.61         0.42   blue      5
    2020-05-15 13:15:00         32.50        47.01         0.49   blue      5
    2020-05-15 13:30:00         32.52        46.67         0.57   blue      5
    2020-05-15 13:45:00         32.68        47.52         0.56   blue      5

                         year  day_of_week  day_of_month  hour  minute  second
    2020-05-15 12:00:00  2020            4            15    12       0       0
    2020-05-15 12:15:00  2020            4            15    12      15       0
    2020-05-15 12:30:00  2020            4            15    12      30       0
    2020-05-15 12:45:00  2020            4            15    12      45       0
    2020-05-15 13:00:00  2020            4            15    13       0       0
    2020-05-15 13:15:00  2020            4            15    13      15       0
    2020-05-15 13:30:00  2020            4            15    13      30       0
    2020-05-15 13:45:00  2020            4            15    13      45       0

We can obtain the name of all the variables in the output dataframe as follows:

.. code:: python

    dtf.get_feature_names_out()

.. code:: python

    ['ambient_temp',
     'module_temp',
     'irradiation',
     'color',
     'month',
     'year',
     'day_of_week',
     'day_of_month',
     'hour',
     'minute',
     'second']


Important
---------

We highly recommend specifying the date and time features that you would like to extract
from your datetime variables.

If you have too many time variables, this might not be possible. In this case, keep in
mind that if you extract date features from variables that have only time, or time features
from variables that have only dates, your features will be meaningless.

Let's explore the outcome with an example. We create a dataset with only time variables.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "not_a_dt": ['not', 'a', 'date', 'time'],
        "var_time1": ['12:34:45', '23:01:02', '11:59:21', '08:44:23'],
        "var_time2": ['02:27:26', '10:10:55', '17:30:00', '18:11:18'],
    })

And now we mistakenly extract only date features:

.. code:: python

    dfts = DatetimeFeatures(
        features_to_extract=["year", "month", "day_of_week"],
        format="mixed",
    )
    df_transf = dfts.fit_transform(toy_df)

    df_transf

.. code:: python

      not_a_dt  var_time1_year  var_time1_month  var_time1_day_of_week  var_time2_year \
    0      not            2021               12                      2            2021
    1        a            2021               12                      2            2021
    2     date            2021               12                      2            2021
    3     time            2021               12                      2            2021

       var_time2_month  var_time2_day_of_week
    0               12                      2
    1               12                      2
    2               12                      2
    3               12                      2

The transformer will still create features derived from today's date (the date of
creating the docs).

If instead we have a dataframe with only date variables:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_date1": ['May-1989', 'Dec-2020', 'Jan-1999', 'Feb-2002'],
        "var_date2": ['06/21/12', '02/10/98', '08/03/10', '10/31/20'],
    })

And we mistakenly extract the hour and the minute:

.. code:: python

    dfts = DatetimeFeatures(
        features_to_extract=["hour", "minute"],
        format="mixed",
    )
    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

       var_date1_hour  var_date1_minute  var_date2_hour  var_date2_minute
    0               0                 0               0                 0
    1               0                 0               0                 0
    2               0                 0               0                 0
    3               0                 0               0                 0

The new features will contain the value 0.

Automating feature extraction
-----------------------------

We can indicate which features we want to extract from the datetime variables as we did
in the previous examples, by passing the feature names in lists.

Alternatively, :class:`DatetimeFeatures()` has default options to extract a group of
commonly used features, or all supported features.

Let's first create a toy dataframe:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_dt1": pd.date_range("2018-01-01", periods=3, freq="H"),
        "var_dt2": ['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21'],
        "var_dt3": ['03/02/15 02:27:26', '02/28/97 10:10:55', '11/11/03 17:30:00'],
    })

Most common features
~~~~~~~~~~~~~~~~~~~~

Now, we will extract the **most common** date and time features from one of the variables.
To do this, we leave the parameter `features_to_extract` to `None`.

.. code:: python

    dfts = DatetimeFeatures(
        variables=["var_dt1"],
        features_to_extract=None,
        drop_original=False,
    )

    df_transf = dfts.fit_transform(toy_df)

    df_transf

.. code:: python

                  var_dt1            var_dt2            var_dt3  var_dt1_month  \
    0 2018-01-01 00:00:00  08/31/00 12:34:45  03/02/15 02:27:26              1
    1 2018-01-01 01:00:00  12/01/90 23:01:02  02/28/97 10:10:55              1
    2 2018-01-01 02:00:00  04/25/01 11:59:21  11/11/03 17:30:00              1

       var_dt1_year  var_dt1_day_of_week  var_dt1_day_of_month  var_dt1_hour  \
    0          2018                    0                     1             0
    1          2018                    0                                   1
    2          2018                    0                  1                2

        var_dt1_minute    var_dt1_second
    0               0                  0
    1               0                  0
    2               0                  0

Our new dataset contains the original features plus the new variables extracted
from them.

We can find the group of features extracted by the transformer in its attribute:

.. code:: python

    dfts.features_to_extract_

.. code:: python

    ['month',
     'year',
     'day_of_week',
     'day_of_month',
     'hour',
     'minute',
     'second']

All supported features
~~~~~~~~~~~~~~~~~~~~~~

We can also extract all supported features automatically, by setting the parameter
`features_to_extract` to `"all"`:

.. code:: python

    dfts = DatetimeFeatures(
        variables=["var_dt1"],
        features_to_extract='all',
        drop_original=False,
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

                  var_dt1            var_dt2            var_dt3  var_dt1_month  \
    0 2018-01-01 00:00:00  08/31/00 12:34:45  03/02/15 02:27:26              1
    1 2018-01-01 01:00:00  12/01/90 23:01:02  02/28/97 10:10:55              1
    2 2018-01-01 02:00:00  04/25/01 11:59:21  11/11/03 17:30:00              1

       var_dt1_quarter  var_dt1_semester  var_dt1_year  \
    0                1                 1          2018
    1                1                 1          2018
    2                1                 1          2018

       var_dt1_week  var_dt1_day_of_week  ...  var_dt1_month_end  var_dt1_quarter_start  \
    0             1                    0  ...                  0                      1
    1             1                    0  ...                  0                      1
    2             1                    0  ...                  0                      1

       var_dt1_quarter_end  var_dt1_year_start  var_dt1_year_end  \
    0                    0                   1                 0
    1                    0                   1                 0
    2                    0                   1                 0

       var_dt1_leap_year  var_dt1_days_in_month  var_dt1_hour  var_dt1_minute  \
    0                  0                     31             0               0
    1                  0                     31             1               0
    2                  0                     31             2               0

       var_dt1_second
    0               0
    1               0
    2               0

We can find the group of features extracted by the transformer in its attribute:

.. code:: python

    dfts.features_to_extract_

.. code:: python

    ['month',
     'quarter',
     'semester',
     'year',
     'week',
     'day_of_week',
     'day_of_month',
     'day_of_year',
     'weekend',
     'month_start',
     'month_end',
     'quarter_start',
     'quarter_end',
     'year_start',
     'year_end',
     'leap_year',
     'days_in_month',
     'hour',
     'minute',
     'second']

Extract and select features automatically
-----------------------------------------

If we have a dataframe with date variables, time variables and date and time variables,
we can extract all features, or the most common features from all the variables, and then
go ahead and remove the irrelevant features with the `DropConstantFeatures()` class.

Let's create a dataframe with a mix of datetime variables.

.. code:: python

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from feature_engine.datetime import DatetimeFeatures
    from feature_engine.selection import DropConstantFeatures

    toy_df = pd.DataFrame({
        "var_date": ['06/21/12', '02/10/98', '08/03/10', '10/31/20'],
        "var_time1": ['12:34:45', '23:01:02', '11:59:21', '08:44:23'],
        "var_dt": ['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21', '04/25/01 11:59:21'],
    })

Now, we line up in a Scikit-learn pipeline the :class:`DatetimeFeatures` and the
`DropConstantFeatures()`. The :class:`DatetimeFeatures` will create date features
derived from today for the time variable, and time features with the value 0 for the
date only variable. `DropConstantFeatures()` will identify and remove these features
from the dataset.

.. code:: python

    pipe = Pipeline([
        ('datetime', DatetimeFeatures(format="mixed")),
        ('drop_constant', DropConstantFeatures()),
    ])

    pipe.fit(toy_df)

.. code:: python

    Pipeline(steps=[('datetime', DatetimeFeatures()),
                    ('drop_constant', DropConstantFeatures())])

.. code:: python

    df_transf = pipe.transform(toy_df)

    print(df_transf)

.. code:: python

       var_date_month  var_date_year  var_date_day_of_week  var_date_day_of_month  \
    0               6           2012                     3                     21
    1               2           1998                     1                     10
    2               8           2010                     1                      3
    3              10           2020                     5                     31

       var_time1_hour  var_time1_minute  var_time1_second  var_dt_month  \
    0              12                34                45             8
    1              23                 1                 2            12
    2              11                59                21             4
    3               8                44                23             4

       var_dt_year  var_dt_day_of_week  var_dt_day_of_month  var_dt_hour  \
    0         2000                   3                   31           12
    1         1990                   5                    1           23
    2         2001                   2                   25           11
    3         2001                   2                   25           11

       var_dt_minute   var_dt_second
    0             34              45
    1              1               2
    2             59              21
    3             59              21

As you can see, we do not have the constant features in the transformed dataset.

Working with different timezones
--------------------------------

Time-aware datetime variables can be particularly cumbersome to work with as far
as the format goes. We will briefly show how :class:`DatetimeFeatures()` deals
with such variables in three different scenarios.

**Case 1**: our dataset contains a time-aware variable in object format,
with potentially different timezones across different observations. 
We pass `utc=True` when initializing the transformer to make sure it
converts all data to UTC timezone.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_tz": ['12:34:45+3', '23:01:02-6', '11:59:21-8', '08:44:23Z']
    })

    dfts = DatetimeFeatures(
        features_to_extract=["hour", "minute"],
        drop_original=False,
        utc=True,
        format="mixed",
    )

    df_transf = dfts.fit_transform(toy_df)

    df_transf

.. code:: python

           var_tz  var_tz_hour  var_tz_minute
    0  12:34:45+3            9             34
    1  23:01:02-6            5              1
    2  11:59:21-8           19             59
    3   08:44:23Z            8             44


**Case 2**: our dataset contains a variable that is cast as a localized
datetime in a particular timezone. However, we decide that we want to get all
the datetime information extracted as if it were in UTC timezone.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    var_tz = pd.Series(['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21'])
    var_tz = pd.to_datetime(var_tz, format="mixed")
    var_tz = var_tz.dt.tz_localize("US/eastern")
    var_tz

.. code:: python

    0   2000-08-31 12:34:45-04:00
    1   1990-12-01 23:01:02-05:00
    2   2001-04-25 11:59:21-04:00
    dtype: datetime64[ns, US/Eastern]

We need to pass `utc=True` when initializing the transformer to revert back to the UTC
timezone.

.. code:: python

    toy_df = pd.DataFrame({"var_tz": var_tz})

    dfts = DatetimeFeatures(
        features_to_extract=["day_of_month", "hour"],
        drop_original=False,
        utc=True,
    )

    df_transf = dfts.fit_transform(toy_df)

    df_transf

.. code:: python

                         var_tz  var_tz_day_of_month  var_tz_hour
    0 2000-08-31 12:34:45-04:00                   31           16
    1 1990-12-01 23:01:02-05:00                    2            4
    2 2001-04-25 11:59:21-04:00                   25           15


**Case 3**: given a variable like *var_tz* in the example above, we now want
to extract the features keeping the original timezone localization,
therefore we pass `utc=False` or `None`. In this case, we leave it to `None` which
is the default option.

.. code:: python

    dfts = DatetimeFeatures(
        features_to_extract=["day_of_month", "hour"],
        drop_original=False,
        utc=None,
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

                         var_tz  var_tz_day_of_month  var_tz_hour
    0 2000-08-31 12:34:45-04:00                   31           12
    1 1990-12-01 23:01:02-05:00                    1           23
    2 2001-04-25 11:59:21-04:00                   25           11

Note that the hour extracted from the variable differ in this dataframe respect to the
one obtained in **Case 2**.

Missing timestamps
------------------

:class:`DatetimeFeatures` has the option to ignore missing timestamps, or raise an error
when a missing value is encountered in a datetime variable.


Additional resources
--------------------

You can find an example of how to use :class:`DatetimeFeatures()` with a real dataset in
the following `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/datetime/DatetimeFeatures.ipynb>`_

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