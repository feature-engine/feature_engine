.. _datetime_features:

.. currentmodule:: feature_engine.datetime

DatetimeFeatures
================

:class:`DatetimeFeatures()` extracts several datetime features from datetime
variables. It works with variables whose original dtype is datetime, and also with
object-like and categorical variables, provided that they can be parsed into datetime
format. It *cannot* extract features from numerical variables.

Oftentimes, datasets contain information related to dates and/or times at which an event
occurred. In pandas dataframes, these datetime variables can be cast as datetime or,
more generically, as object.

Datetime variables, in their raw format, are generally not suitable to train machine
learning models. Yet, an enormous amount of information can be extracted from them.

:class:`DatetimeFeatures()` is able to extract many numerical and binary date and
time features from these datetime variables. Among these features we can find the month
in which an event occurred, the day of the week, or whether that day was a weekend day.

With :class:`DatetimeFeatures()` you can choose which date and time features
to extract from your datetime variables. You can also extract date and time features
from a subset of datetime variables in your data.

Examples
--------
Through the following examples we highlight the functionality and versatility of
:class:`DatetimeFeatures()`.

Extract date features
~~~~~~~~~~~~~~~~~~~~~

In this example, we are going to extract three date features from a
specific variable in the dataframe. In particular, we are interested
in the month, the day of the year, and whether that day was the last
day of its correspondent month.

First, we will create a toy dataframe with 2 date variables.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_date1": ['May-1989', 'Dec-2020', 'Jan-1999', 'Feb-2002'],
        "var_date2": ['06/21/12', '02/10/98', '08/03/10', '10/31/20'],
    })

Now, we will extract the variables month, month-end and the day of the year from the
second datetime variable in our dataset.

.. code:: python

    dtfs = DatetimeFeatures(
        variables="var_date2",
        features_to_extract=["month", "month_end", "day_of_the_year"]
    )

    df_transf = dtfs.fit_transform(toy_df)

    print(df_transf)

.. code:: python

      var_date1  var_date2_month  var_date2_month_end  var_date2_doty
    0  May-1989                6                    0             173
    1  Dec-2020                2                    0              41
    2  Jan-1999                8                    0             215
    3  Feb-2002               10                    1             305


With `transform()`, the features extracted from the datetime variable are added to the
dataframe.

By default, :class:`DatetimeFeatures()` drops the variable from which the date and time
features were extracted, in this case, *var_date2*. To keep the variable, we just need
to indicate `drop_original=False` when initializing the transformer.

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

.. code:: python

    dfts = DatetimeFeatures(features_to_extract=["minute"])

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

      not_a_dt  var_time1_minute  var_time2_minute
    0      not                34                27
    1        a                 1                10
    2     date                59                30
    3     time                44                11


The transformer found two variales in the dataframe that can be cast to datetime and
proceeded to extract the requested feature from them.

We can find the variables detected as datetime by the transformer in one of its
attributes.

.. code:: python

    dfts.variables_

.. code:: python

    ['var_time1', 'var_time2']

Again, the original datetime variables are dropped from the data by default. If we
want to keep them, we just need to indicate `drop_original=False` when initializing
the transformer.

Extract date and time features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will combine what we have seen in the previous two examples
and extract a date feature - *year* - and time feature - *hour* -
from two variables that contain both date and time information.

Let's go ahead and create a toy dataset.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_dt1": pd.date_range("2018-01-01", periods=3, freq="H"),
        "var_dt2": ['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21'],
        "var_dt3": ['03/02/15 02:27:26', '02/28/97 10:10:55', '11/11/03 17:30:00'],
    })

Now, we set up the :class:`DatetimeFeatures()` to extract features only from 2 of the
variables. In this case, we do not want to drop the datetime variable after extracting
the features.

.. code:: python

    dfts = DatetimeFeatures(
        variables=["var_dt1", "var_dt3"],
        features_to_extract=["year", "hour"],
        drop_original=False,
    )
    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

                  var_dt1            var_dt2            var_dt3  var_dt1_year  \
    0 2018-01-01 00:00:00  08/31/00 12:34:45  03/02/15 02:27:26          2018
    1 2018-01-01 01:00:00  12/01/90 23:01:02  02/28/97 10:10:55          2018
    2 2018-01-01 02:00:00  04/25/01 11:59:21  11/11/03 17:30:00          2018

       var_dt1_hour  var_dt3_year  var_dt3_hour
    0             0          2015             2
    1             1          1997            10
    2             2          2003            17

And that is it. The additional features are now added in the dataframe.

Important
~~~~~~~~~

We highly recommend specifying the date and time features that you would like to extract
from your datetime variables. If you have too many time variables, this might not be
possible. In this case, keep in mind that if you extract date features from variables
that have only time, or time features from variables that have only dates, your features
will be meaningless.

Let's explore the outcome with an example. We create a dataset with only time variables.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "not_a_dt": ['not', 'a', 'date', 'time'],
        "var_time1": ['12:34:45', '23:01:02', '11:59:21', '08:44:23'],
        "var_time2": ['02:27:26', '10:10:55', '17:30:00', '18:11:18'],
    })

And now we mistakenly extract only date features.

.. code:: python

    dfts = DatetimeFeatures(
        features_to_extract=["year", "month", "day_of_the_week"],
    )
    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

      not_a_dt  var_time1_year  var_time1_month  var_time1_dotw  var_time2_year  \
    0      not            2021               12               2            2021
    1        a            2021               12               2            2021
    2     date            2021               12               2            2021
    3     time            2021               12               2            2021

       var_time2_month  var_time2_dotw
    0               12               2
    1               12               2
    2               12               2
    3               12               2

The transformer will still create features derived from today's date (the date of
creating the docs).

If instead we have a dataframe with only date variables.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_date1": ['May-1989', 'Dec-2020', 'Jan-1999', 'Feb-2002'],
        "var_date2": ['06/21/12', '02/10/98', '08/03/10', '10/31/20'],
    })

And mistakenly extract the hour and the minute.

.. code:: python

    dfts = DatetimeFeatures(
        features_to_extract=["hour", "minute"],
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can indicate which features we want to extract from the datetime variables as we did
in the previous examples, passing the feature names in lists. Alternatively,
:class:`DatetimeFeatures()` has default options to extract a group of commonly used
features, or all supported features.

Let's first create a toy dataframe.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_dt1": pd.date_range("2018-01-01", periods=3, freq="H"),
        "var_dt2": ['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21'],
        "var_dt3": ['03/02/15 02:27:26', '02/28/97 10:10:55', '11/11/03 17:30:00'],
    })

Now, we will extract the most common date and time features from one of the variables.
To do this, we leave the parameter `features_to_extract` to `None`.

.. code:: python

    dfts = DatetimeFeatures(
        variables=["var_dt1"],
        features_to_extract=None,
        drop_original=False,
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

                  var_dt1            var_dt2            var_dt3  var_dt1_month  \
    0 2018-01-01 00:00:00  08/31/00 12:34:45  03/02/15 02:27:26              1
    1 2018-01-01 01:00:00  12/01/90 23:01:02  02/28/97 10:10:55              1
    2 2018-01-01 02:00:00  04/25/01 11:59:21  11/11/03 17:30:00              1

       var_dt1_year  var_dt1_dotw  var_dt1_dotm  var_dt1_hour  var_dt1_minute  \
    0          2018             0             1             0               0
    1          2018             0             1             1               0
    2          2018             0             1             2               0

       var_dt1_second
    0               0
    1               0
    2               0

Our new dataset contains the original features plus the new variables extracted
from them.

We can find the group of features extracted by the transformer in its attribute.

.. code:: python

    dfts.features_to_extract_

.. code:: python

    ['month',
     'year',
     'day_of_the_week',
     'day_of_the_month',
     'hour',
     'minute',
     'second']

We can also extract all supported features automatically.

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

       var_dt1_quarter  var_dt1_semester  var_dt1_year  var_dt1_wotm  \
    0                1                 1          2018             1
    1                1                 1          2018             1
    2                1                 1          2018             1

       var_dt1_woty  var_dt1_dotw  ...  var_dt1_month_end  var_dt1_quarter_start  \
    0             1             0  ...                  0                      1
    1             1             0  ...                  0                      1
    2             1             0  ...                  0                      1

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

We can find the group of features extracted by the transformer in its attribute.

.. code:: python

    dfts.features_to_extract_

.. code:: python

    ['month',
     'quarter',
     'semester',
     'year',
     'week_of_the_month',
     'week_of_the_year',
     'day_of_the_week',
     'day_of_the_month',
     'day_of_the_year',
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we have a dataframe with a date variables, time variables and date and time variables,
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

Now, we align in a Scikit-learn pipeline the :class:`DatetimeFeatures` and the
`DropConstantFeatures()`. The :class:`DatetimeFeatures` will create date features
derived from today for the time variable, and time features with the value 0 for the
date only variable. `DropConstantFeatures()` will identify and remove these features
from the dataset.

.. code:: python

    pipe = Pipeline([
        ('datetime', DatetimeFeatures()),
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

       var_date_month  var_date_year  var_date_dotw  var_date_dotm  \
    0               6           2012              3             21
    1               2           1998              1             10
    2               8           2010              1              3
    3              10           2020              5             31

       var_time1_hour  var_time1_minute  var_time1_second  var_dt_month  \
    0              12                34                45             8
    1              23                 1                 2            12
    2              11                59                21             4
    3               8                44                23             4

       var_dt_year  var_dt_dotw  var_dt_dotm  var_dt_hour  var_dt_minute  \
    0         2000            3           31           12             34
    1         1990            5            1           23              1
    2         2001            2           25           11             59
    3         2001            2           25           11             59

       var_dt_second
    0             45
    1              2
    2             21
    3             21

As you can see, we do not have the constant features in the transformed dataset.

Extract features from time-aware variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        utc=True
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

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
    var_tz = pd.to_datetime(var_tz)
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
        features_to_extract=["day_of_the_month", "hour"],
        drop_original=False,
        utc=True,
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

                         var_tz  var_tz_dotm  var_tz_hour
    0 2000-08-31 12:34:45-04:00           31           16
    1 1990-12-01 23:01:02-05:00            2            4
    2 2001-04-25 11:59:21-04:00           25           15


**Case 3**: given a variable like *var_tz* in the example above, we now want
to extract the features keeping the original timezone localization,
therefore we pass `utc=False` or `None`. In this case, we leave it to `None` which
is the default option.

.. code:: python

    dfts = DatetimeFeatures(
        features_to_extract=["day_of_the_month", "hour"],
        drop_original=False,
        utc=None,
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf)

.. code:: python

                         var_tz  var_tz_dotm  var_tz_hour
    0 2000-08-31 12:34:45-04:00           31           12
    1 1990-12-01 23:01:02-05:00            1           23
    2 2001-04-25 11:59:21-04:00           25           11

Note that the hour extracted from the variable differ in this dataframe respect to the
one obtained in **Case 2**.

More details
^^^^^^^^^^^^

You can find additional examples with a real dataset on how to use
:class:`DatetimeFeatures()` in the following Jupyter notebook.

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/datetime/DatetimeFeatures.ipynb>`_
