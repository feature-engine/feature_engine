.. _datetime_ordinal:

.. currentmodule:: feature_engine.datetime


DatetimeOrdinal
===============

:class:`DatetimeOrdinal()` converts datetime variables into ordinal numbers, thereby
providing a numerical representation of the date. By default, it returns the proleptic
Gregorian ordinal of the date, where 1st January of year 1 has ordinal 1.

If 1st January of year 1 has ordinal number 1 then, 2nd January of year 1 will have ordinal
number 2, and so on.

Optionally, :class:`DatetimeOrdinal()` can compute the number of days relative to a
user-defined `start_date`. This can be useful for reducing the magnitude of the ordinal
values and for aligning them to a specific project timeline.

Ordinal numbers preserve the relative distances between dates (e.g., the number of days
between events), allowing algorithms to capture linear trends and temporal distances.


Datetime ordinals with pandas
-----------------------------

In Python, we can get the Gregorian ordinal of a date using the `toordinal()` method
from a datetime object as follows:

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

In the variable `ordinal`, the value `738521` means that `2023-01-01` is 738521 days
*after* the 1st of January of the year 1.

Datetime ordinal with feature-engine
------------------------------------

:class:`DatetimeOrdinal()` automatically converts one or more datetime variables into
ordinal numbers. It works with variables whose dtype is datetime, as well as with
object-type variables, provided that they can be parsed into datetime format.

:class:`DatetimeOrdinal()` uses pandas `toordinal()` under the hood. The main
functionalities are:

- It can convert multiple datetime variables at once.
- It can compute the ordinal number relative to a `start_date`.
- It can automatically find and select datetime variables.

.. attention::

    **New in version 2.0:** When `variables` is `None`, :class:`DatetimeOrdinal()` used to
    raise an error if the dataframe contained no datetime variables. You can now set the new
    parameter `return_empty` to `True` to make the transformer return an empty list of
    variables and skip the transformation instead, leaving the dataframe unchanged. This
    lets you reuse the same pipeline across different datasets or projects, some of which
    may not contain datetime variables, without building a tailored pipeline for each one.
    `return_empty` will default to `True` from version 2.1 onwards.

Python implementation
---------------------

First, let's create a toy dataframe with 2 date variables:

.. code:: python

    import pandas as pd
    from feature_engine.datetime import DatetimeOrdinal

    toy_df = pd.DataFrame({
        "var_date1": ['May-1989', 'Dec-2020', 'Jan-1999', 'Feb-2002'],
        "var_date2": ['06/21/2012', '02/10/1998', '08/03/2010', '10/31/2020'],
        "other_var": [1, 2, 3, 4]
    })

Now, we will set up the transformer to convert `var_date2` into an ordinal feature:

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

By default, :class:`DatetimeOrdinal()` drops the original datetime variable. To keep
it, you can set `drop_original=False`.

Calculate days from a start date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`DatetimeOrdinal()` can also calculate the number of days elapsed since a
specific `start_date` as follows:

.. code:: python

    dtfs = DatetimeOrdinal(
        variables="var_date2",
        start_date="2010-01-01"
    )

    df_transf = dtfs.fit_transform(toy_df)

    df_transf

The new feature now represents the number of days between `var_date2` and January 1st,
2010. Note that dates before the `start_date` will result in negative numbers:

.. code:: python

      var_date1  other_var  var_date2_ordinal
    0    May-1989          1                903
    1    Dec-2020          2              -4343
    2    Jan-1999          3                215
    3    Feb-2002          4               3956


Missing timestamps
------------------

:class:`DatetimeOrdinal()` handles missing values (NaT) in datetime variables through
the `missing_values` parameter, which can be set to `"raise"` or `"ignore"`.

If `missing_values="raise"`, the transformer will raise an error if NaT values are
found in the datetime variables during `fit()` or `transform()`.

If `missing_values="ignore"`, the transformer will ignore NaT values, and the resulting
ordinal feature will contain `NaN` (or `pd.NA`) in their place.


Additional resources
--------------------

For tutorials about this and other feature engineering methods check out these resources:

- `Feature Engineering for Machine Learning <https://www.trainindata.com/p/feature-engineering-for-machine-learning>`_, online course.
- `Feature Engineering for Time Series Forecasting <https://www.trainindata.com/p/feature-engineering-for-forecasting>`_, online course.
- `Python Feature Engineering Cookbook <https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587>`_, book.

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting `Sole <https://linkedin.com/in/soledad-galli>`_,
the main developer of feature-engine.