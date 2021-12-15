.. _datetime_features:

.. currentmodule:: feature_engine.datetime

DatetimeFeatures
================

The :class:`DatetimeFeatures()` extracts several datetime features from datetime
variables. It works with variables whose original dtype is datetime, and also with
object-like and categorical variables, provided that they can be parsed into datetime
format. It *cannot* extract features from numerical variables.

Oftentimes datasets contain information related dates and/or times at which an event
occurred. In pandas dataframes, these datetime variables can be cast as datetime or,
more generically, as object.

Datetime variables in their raw format, are generally not suitable to train machine
learning models. Yet, an enormous amount of information can be extracted from them.

The :class:`DatetimeFeatures()` is able to extract many numerical and binary date and
time features from these datetime variables. Among these features we can find the month
in which an event occurred, the day of the week, or whether that day was a weekend day.

With the The :class:`DatetimeFeatures()` you can choose which date and time features
to extract from your datetime variables. You can also extract date and time features
from a subset of datetime variables in your data.

Examples
--------

Extract date features
~~~~~~~~~~~~~~~~~~~~~

In the following example we are going to extract three date features from a
specific column in the dataframe. In particular, we are interested
in the month, the day of the year, and whether that day was the last
day of its correspondent month. All these features, including the binary one,
will be cast as integer.

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeFeatures

    toy_df = pd.DataFrame({
        "var_date1": ['May-1989', 'Dec-2020', 'Jan-1999', 'Feb-2002'],
        "var_date2": ['06/21/12', '02/10/98', '08/03/10', '10/31/20'],
    })

    dtfs = DatetimeFeatures(
        variables="var_date2",
        features_to_extract=["month", "month_end", "day_of_the_year"]
    )

    df_transf = dtfs.fit_transform(toy_df)

    print(df_transf.to_markdown(tablefmt="pipe"))

.. code:: markdown

    |    | var_date1   |   var_date2_month |   var_date2_month_end |   var_date2_doty |
    |---:|:------------|------------------:|----------------------:|-----------------:|
    |  0 | May-1989    |                 6 |                     0 |              173 |
    |  1 | Dec-2020    |                 2 |                     0 |               41 |
    |  2 | Jan-1999    |                 8 |                     0 |              215 |
    |  3 | Feb-2002    |                10 |                     1 |              305 |

Note: the column *var_date2* in its original format is not present
in the transformed dataframe anymore. If we wish to keep it, we just need to pass
the argument drop_original=False when initializing the transformer

Extract time features
~~~~~~~~~~~~~~~~~~~~~

In the following example we are going to extract the *minute* feature
from the two datetime columns in our dataset, without specifying them.

.. code:: python 

    toy_df = pd.DataFrame({
        "not_a_dt": ['not', 'a', 'date', 'time'],
        "var_time1": ['12:34:45', '23:01:02', '11:59:21', '08:44:23'],
        "var_time2": ['02:27:26', '10:10:55', '17:30:00', '18:11:18'],
    })

    dfts = DatetimeFeatures(features_to_extract=["minute"])

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf.to_markdown(tablefmt="pipe"))

.. code:: markdown

    |    | not_a_dt   |   var_time1_minute |   var_time2_minute |
    |---:|:-----------|-------------------:|-------------------:|
    |  0 | not        |                 34 |                 27 |
    |  1 | a          |                  1 |                 10 |
    |  2 | date       |                 59 |                 30 |
    |  3 | time       |                 44 |                 11 |

The transformer found two columns in the dataframe that could be cast to datetime and
proceeded to extract the requested feature from them.

Extract date and time features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following example we will combine what we have seen in the previous
ones and extract a date feature - *year* - and time feature - *hour* -
from two columns that contain both date and time information

.. code:: python

    toy_df = pd.DataFrame({
        "var_dt1": pd.date_range("2018-01-01", periods=3, freq="H"),
        "var_dt2": ['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21'],
        "var_dt3": ['03/02/15 02:27:26', '02/28/97 10:10:55', '11/11/03 17:30:00'],
    })

    dfts = DatetimeFeatures(
        variables=["var_dt1", "var_dt3"],
        features_to_extract=["year", "hour"]
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf.to_markdown(tablefmt="pipe"))

.. code:: markdown

    |    | var_dt2           |   var_dt1_year |   var_dt1_hour |   var_dt3_year |   var_dt3_hour |
    |---:|:------------------|---------------:|---------------:|---------------:|---------------:|
    |  0 | 08/31/00 12:34:45 |           2018 |              0 |           2015 |              2 |
    |  1 | 12/01/90 23:01:02 |           2018 |              1 |           1997 |             10 |
    |  2 | 04/25/01 11:59:21 |           2018 |              2 |           2003 |             17 |

Extract features from time-aware variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Time-aware datetime variables can be particularly cumbersome to work with as far
as the format goes. We will briefly show how :class:`DatetimeFeatures()` deals
with such variables in three different scenarios.

Case 1: our dataset contains a time-aware variable in object format,
with potentially different timezones across different observations. 
We pass utc=True when initializing the transformer to make sure it
converts all data to UTC timezone.

.. code:: python
    
    toy_df = pd.DataFrame({
        "var_tz": ['12:34:45+3', '23:01:02-6', '11:59:21-8', '08:44:23Z']
    })

    dfts = DatetimeFeatures(
        features_to_extract=["hour", "minute"],
        drop_original=False,
        utc=True
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf.to_markdown(tablefmt="pipe"))

.. code:: markdown

    |    | var_tz     |   var_tz_hour |   var_tz_minute |
    |---:|:-----------|--------------:|----------------:|
    |  0 | 12:34:45+3 |             9 |              34 |
    |  1 | 23:01:02-6 |             5 |               1 |
    |  2 | 11:59:21-8 |            19 |              59 |
    |  3 | 08:44:23Z  |             8 |              44 |

Case 2: our dataset contains a variable that is cast as a localized
datetime in a particular timezone. However, we decide that we want to get all
the datetime information extracted as if it were in UTC timezone.

.. code:: python

    var_tz = pd.Series(['08/31/00 12:34:45', '12/01/90 23:01:02', '04/25/01 11:59:21'])
    var_tz = pd.to_datetime(var_tz)
    var_tz = var_tz.dt.tz_localize("US/eastern")
    var_tz

.. code:: markdown

    0   2000-08-31 12:34:45-04:00
    1   1990-12-01 23:01:02-05:00
    2   2001-04-25 11:59:21-04:00
    dtype: datetime64[ns, US/Eastern]

.. code:: python

    toy_df = pd.DataFrame({"var_tz": var_tz})

    dfts = DatetimeFeatures(
        features_to_extract=["day_of_the_month", "hour"],
        drop_original=False,
        utc=True
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf.to_markdown(tablefmt="pipe"))

.. code:: markdown

    |    | var_tz                    |   var_tz_dotm |   var_tz_hour |
    |---:|:--------------------------|--------------:|--------------:|
    |  0 | 2000-08-31 12:34:45-04:00 |            31 |            16 |
    |  1 | 1990-12-01 23:01:02-05:00 |             2 |             4 |
    |  2 | 2001-04-25 11:59:21-04:00 |            25 |            15 |

Case 3: given a variable like *var_tz* in the example above, we now want
to extract the features keeping the original timezone localization,
therefore we pass utc=False (which is the default option).

.. code:: python

    dfts = DatetimeFeatures(
        features_to_extract=["day_of_the_month", "hour"],
        drop_original=False,
    )

    df_transf = dfts.fit_transform(toy_df)

    print(df_transf.to_markdown(tablefmt="pipe"))

.. code:: markdown

    |    | var_tz                    |   var_tz_dotm |   var_tz_hour |
    |---:|:--------------------------|--------------:|--------------:|
    |  0 | 2000-08-31 12:34:45-04:00 |            31 |            12 |
    |  1 | 1990-12-01 23:01:02-05:00 |             1 |            23 |
    |  2 | 2001-04-25 11:59:21-04:00 |            25 |            11 |

More details
^^^^^^^^^^^^

You can find creative ways to use the :class:`DatetimeFeatures()` in the
following Jupyter notebook.

#TODO: change the link below to your notebook:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/creation/MathematicalCombination.ipynb>`_
