.. _datetime_ordinal:

.. currentmodule:: feature_engine.datetime

DatetimeOrdinal
================

:class:`DatetimeOrdinal()` converts datetime variables into ordinal numbers, that is, a numerical representation of the date.

Datetime variables cannot be used directly by machine learning algorithms because they are not numerical. However, they contain valuable information about sequences of events or elapsed time. 

By converting datetime variables into ordinal numbers, we can capture this information while discarding the complexities of raw datetime formats. Ordinal numbers preserve the relative distances between dates (e.g., the number of days between events), allowing algorithms to capture linear trends, calculate temporal distances naturally, and handle time consistently without needing to parse or split the datetime into multiple separate features like year, month, or day.

By default, :class:`DatetimeOrdinal()` returns the proleptic Gregorian ordinal, where January 1 of year 1 has ordinal 1.

Optionally, :class:`DatetimeOrdinal()` can compute the number of days relative to a user-defined `start_date`.

Datetime ordinals with pandas
-----------------------------

In Python, we can get the Gregorian ordinal of a date using the `toordinal()` method from a datetime object.

.. code:: python

    import pandas as pd

    data = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-01-10"])})

    data["ordinal"] = data["date"].apply(lambda x: x.toordinal())

    data

The output shows the new ordinal feature:

.. code:: python

        date   ordinal
    0 2023-01-01    738521
    1 2023-01-10    738530


Datetime ordinals with Feature-engine
-------------------------------------

:class:`DatetimeOrdinal()` automatically converts one or more datetime variables into ordinal numbers. It works with variables whose dtype is datetime, as well as with object-type variables, provided that they can be parsed into datetime format.

:class:`DatetimeOrdinal()` uses pandas `toordinal()` under the hood. The main functionalities are:

- It can convert multiple datetime variables at once.
- It can compute the ordinal number relative to a `start_date`.
- It can automatically find and select datetime variables.

Example
~~~~~~~

First, let's create a toy dataframe with 2 date variables:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeOrdinal

    toy_df = pd.DataFrame({
        "var_date1": ['May-1989', 'Dec-2020', 'Jan-1999', 'Feb-2002'],
        "var_date2": ['06/21/2012', '02/10/1998', '08/03/2010', '10/31/2020'],
        "other_var": [1, 2, 3, 4]
    })

Now, we will set up the transformer to convert `var_date2` into an ordinal feature.

.. code:: python

    dtfs = DatetimeOrdinal(variables="var_date2")

    df_transf = dtfs.fit_transform(toy_df)

    df_transf

We see the new ordinal feature in the output:

.. code:: python

      var_date1  other_var  var_date2_ordinal
    0    May-1989          1             734675
    1    Dec-2020          2             729430
    2    Jan-1999          3             733987
    3    Feb-2002          4             737729

By default, :class:`DatetimeOrdinal()` drops the original datetime variable. To keep it, you can set `drop_original=False`.

Calculate days from a start date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`DatetimeOrdinal()` can also calculate the number of days elapsed since a specific `start_date`.

.. code:: python

    dtfs = DatetimeOrdinal(
        variables="var_date2",
        start_date="2010-01-01"
    )

    df_transf = dtfs.fit_transform(toy_df)

    df_transf

The new feature now represents the number of days between `var_date2` and January 1st, 2010. Note that dates before the `start_date` will result in negative numbers.

.. code:: python

      var_date1  other_var  var_date2_ordinal
    0    May-1989          1                903
    1    Dec-2020          2              -4343
    2    Jan-1999          3                215
    3    Feb-2002          4               3956


Missing timestamps
------------------

:class:`DatetimeOrdinal()` handles missing values (NaT) in datetime variables through the `missing_values` parameter, which can be set to `"raise"` or `"ignore"`.

If `missing_values="raise"`, the transformer will raise an error if NaT values are found in the datetime variables during `fit()` or `transform()`.

If `missing_values="ignore"`, the transformer will ignore NaT values, and the resulting ordinal feature will contain `NaN` (or `pd.NA`) in their place.


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