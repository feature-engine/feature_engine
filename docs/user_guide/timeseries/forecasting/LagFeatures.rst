.. _lag_features:

.. currentmodule:: feature_engine.timeseries.forecasting

LagFeatures
===========

The :class:`LagFeatures` adds lag features to the dataframe. A lag feature is a feature
with information about a prior time step.

When forecasting the future values of a variable the past values of that variable are
likely to be predictive. Past values of other accompanying predictive features can also
help us improve our forecast. Thus, in forecasting, it is common practice to create lag
features from target series and predictive variables.

A lag feature is the target or feature k period(s) in the past, where k is the lag and
is to be set by the user. We can create features with multiple lags by varying k. There
are 2 ways in which we can indicate the lag k using :class:`LagFeatures`. We can
indicate the lag using the parameter `periods`. This parameter takes integers that
indicate the number of rows forward that the features will be lagged. Alternatively,
we can use the parameter `freq`, which takes a string with the period and frequency,
and lags features based on the datetime index. For example, if we pass `freq="1D"`, the
values of the features will be moved 1 day forward.

The :class:`LagFeatures` transformer works very similarly to `pandas.shift`, but unlike
`pandas.shift` we can indicate the lag using either `periods` or `freq` but not both.
Also, unlike `pandas.shift`, we can only lag features forward. The :class:`LagFeatures`
has several advantages over `pandas.shift` however. First, it can create features with
multiple values of k at the same time. Second, it adds the features with a name to the
original dataframe. Third, it has the methods `fit()` and `transform()` that make it
compatible with the Scikit-learn's `Pipeline` and cross-validation functions.

Examples
--------

Let's create a toy dataset to demonstrate the functionality of :class:`LagFeatures`.
The dataframe contains 3 numerical variables a categorical variable and a datetime
index.

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

Shift a row forward
~~~~~~~~~~~~~~~~~~~

Now we will create new features by lagging all numerical variables 1 row forward. Note
that :class:`LagFeatures` automatically finds all numerical variables.

.. code:: python

    from feature_engine.timeseries.forecasting import LagFeatures

    lag_f = LagFeatures(periods=1)

    X_tr = lag_f.fit_transform(X)

    X_tr.head()

We can find the lag features on the right side of the dataframe. Note that the values
have been shifted a row forward.

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                         ambient_temp_lag_1  module_temp_lag_1  irradiation_lag_1
    2020-05-15 12:00:00                 NaN                NaN                NaN
    2020-05-15 12:15:00               31.31              49.18               0.51
    2020-05-15 12:30:00               31.51              49.84               0.79
    2020-05-15 12:45:00               32.15              52.35               0.65
    2020-05-15 13:00:00               32.39              50.63               0.76

The variables to lag are stored in the `variables_` attribute of the
:class:`LagFeatures`:

.. code:: python

    lag_f.variables_

.. code:: python

    ['ambient_temp', 'module_temp', 'irradiation']

We can obtain the names of the variables in the returned dataframe using the
get_feature_names_out() method:

.. code:: python

    lag_f.get_feature_names_out()

.. code:: python

    ['ambient_temp',
     'module_temp',
     'irradiation',
     'color',
     'ambient_temp_lag_1',
     'module_temp_lag_1',
     'irradiation_lag_1']

Create multiple lag features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can create multiple lag features with one transformer by passing the lag periods in
a list.

.. code:: python

    lag_f = LagFeatures(periods=[1, 2])

    X_tr = lag_f.fit_transform(X)

    X_tr.head()

Note how multiple lag features were created for each of the numerical variables and
added at the back of the dataframe.

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                         ambient_temp_lag_1  module_temp_lag_1  irradiation_lag_1  \
    2020-05-15 12:00:00                 NaN                NaN                NaN
    2020-05-15 12:15:00               31.31              49.18               0.51
    2020-05-15 12:30:00               31.51              49.84               0.79
    2020-05-15 12:45:00               32.15              52.35               0.65
    2020-05-15 13:00:00               32.39              50.63               0.76

                         ambient_temp_lag_2  module_temp_lag_2  irradiation_lag_2
    2020-05-15 12:00:00                 NaN                NaN                NaN
    2020-05-15 12:15:00                 NaN                NaN                NaN
    2020-05-15 12:30:00               31.31              49.18               0.51
    2020-05-15 12:45:00               31.51              49.84               0.79
    2020-05-15 13:00:00               32.15              52.35               0.65

We can get the names of all the lag features created for the variable "irradiation" as
follows:

.. code:: python

    lag_f.get_feature_names_out(["irradiation"])

.. code:: python

    ['irradiation_lag_1', 'irradiation_lag_2']


Lag based on datetime
~~~~~~~~~~~~~~~~~~~~~

We can also lag features utilizing information in the datetime index of the dataframe.
Let's for example create features by lagging 2 of the numerical variables 30 minutes
forward.

.. code:: python

    lag_f = LagFeatures(variables = ["module_temp", "irradiation"], freq="30min")

    X_tr = lag_f.fit_transform(X)

    X_tr.head()

Note that the features were moved forward 30 minutes.

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                         module_temp_lag_30min  irradiation_lag_30min
    2020-05-15 12:00:00                    NaN                    NaN
    2020-05-15 12:15:00                    NaN                    NaN
    2020-05-15 12:30:00                  49.18                   0.51
    2020-05-15 12:45:00                  49.84                   0.79
    2020-05-15 13:00:00                  52.35                   0.65

Drop variable after lagging features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, we can lag multiple time intervals forward, but this time, let's drop the
original variable after creating the lag features.

.. code:: python

    lag_f = LagFeatures(variables="irradiation",
                        freq=["30min", "45min"],
                        drop_original=True,
                        )

    X_tr = lag_f.fit_transform(X)

    X_tr.head()

We now see the multiple lag features at the back of the dataframe, and also that the
original variable is not present in the output dataframe.

.. code:: python

                         ambient_temp  module_temp  color  irradiation_lag_30min  \
    2020-05-15 12:00:00         31.31        49.18  green                    NaN
    2020-05-15 12:15:00         31.51        49.84  green                    NaN
    2020-05-15 12:30:00         32.15        52.35  green                   0.51
    2020-05-15 12:45:00         32.39        50.63  green                   0.79
    2020-05-15 13:00:00         32.62        49.61   blue                   0.65

                         irradiation_lag_45min
    2020-05-15 12:00:00                    NaN
    2020-05-15 12:15:00                    NaN
    2020-05-15 12:30:00                    NaN
    2020-05-15 12:45:00                   0.51
    2020-05-15 13:00:00                   0.79

Working with pandas series
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your time series is a pandas Series instead of a pandas Dataframe, you need to
transform it into a dataframe before using :class:`LagFeatures`.

The following is a pandas Series:

.. code:: python

    X['ambient_temp']

.. code:: python

    2020-05-15 12:00:00    31.31
    2020-05-15 12:15:00    31.51
    2020-05-15 12:30:00    32.15
    2020-05-15 12:45:00    32.39
    2020-05-15 13:00:00    32.62
    2020-05-15 13:15:00    32.50
    2020-05-15 13:30:00    32.52
    2020-05-15 13:45:00    32.68
    Freq: 15T, Name: ambient_temp, dtype: float64

We can use :class:`LagFeatures` to create, for example, 3 new features by lagging the
pandas Series if we convert it to a pandas Dataframe using the method `to_frame()`:

.. code:: python

    lag_f = LagFeatures(periods=[1, 2, 3])

    X_tr = lag_f.fit_transform(X['ambient_temp'].to_frame())

    X_tr.head()

.. code:: python

                         ambient_temp  ambient_temp_lag_1  ambient_temp_lag_2  \
    2020-05-15 12:00:00         31.31                 NaN                 NaN
    2020-05-15 12:15:00         31.51               31.31                 NaN
    2020-05-15 12:30:00         32.15               31.51               31.31
    2020-05-15 12:45:00         32.39               32.15               31.51
    2020-05-15 13:00:00         32.62               32.39               32.15

                         ambient_temp_lag_3
    2020-05-15 12:00:00                 NaN
    2020-05-15 12:15:00                 NaN
    2020-05-15 12:30:00                 NaN
    2020-05-15 12:45:00               31.31
    2020-05-15 13:00:00               31.51

And if we do not want the original values of time series in the returned dataframe, we
just need to remember to drop the original series after the transformation:

.. code:: python

    lag_f = LagFeatures(periods=[1, 2, 3], drop_original=True)

    X_tr = lag_f.fit_transform(X['ambient_temp'].to_frame())

    X_tr.head()

.. code:: python

                         ambient_temp_lag_1  ambient_temp_lag_2  \
    2020-05-15 12:00:00                 NaN                 NaN
    2020-05-15 12:15:00               31.31                 NaN
    2020-05-15 12:30:00               31.51               31.31
    2020-05-15 12:45:00               32.15               31.51
    2020-05-15 13:00:00               32.39               32.15

                         ambient_temp_lag_3
    2020-05-15 12:00:00                 NaN
    2020-05-15 12:15:00                 NaN
    2020-05-15 12:30:00                 NaN
    2020-05-15 12:45:00               31.31
    2020-05-15 13:00:00               31.51


Getting the name of new variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can easily obtain the name of the original and new variables with the method
`get_feature_names_out`. By using the method with the default parameters, we obtain
all the features in the output dataframe.

.. code:: python

    lag_f = LagFeatures(periods=[1, 2])

    lag_f.fit(X)

    lag_f.get_feature_names_out()

.. code:: python

    ['ambient_temp',
     'module_temp',
     'irradiation',
     'color',
     'ambient_temp_lag_1',
     'module_temp_lag_1',
     'irradiation_lag_1',
     'ambient_temp_lag_2',
     'module_temp_lag_2',
     'irradiation_lag_2']

Alternatively, we can obtain the names of the lag features created from one or more
input features as follows:

.. code:: python

    lag_f.get_feature_names_out(input_features=["irradiation"])

.. code:: python

    ['irradiation_lag_1', 'irradiation_lag_2']

