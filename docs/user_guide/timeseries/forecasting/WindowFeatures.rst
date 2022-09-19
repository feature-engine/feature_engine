.. _window_features:

.. currentmodule:: feature_engine.timeseries.forecasting

WindowFeatures
==============

:class:`WindowFeatures` adds window features to the dataframe. Window features are
the result of window operations over the variables. Window operations are operations that
perform an aggregation over a sliding partition of past values. A window feature is,
then, a feature created after computing mathematical functions (e.g., mean, min,
max, etc.) within a window over the past data.

For example, the mean value of the previous 3 months of data is a window feature. The
maximum value of the previous three rows of data is another window feature.

When forecasting the future values of a variable, the past values of that variable are
likely to be predictive. To capitalize on the past values of a variable, we can simply
lag features with :class:`LagFeatures`. And, we can as well create features that
take in consideration the values in the past but within a window.

To create window features we need to determine a number of parameters. First, we need
to identify the size of the window or windows in which we will perform the operations.
For example, we can take the average of the variable over 3 months, or 6 weeks. We also
need to determine how far back is the window located respect to the value we want to
forecast. For example, I can take the average of the last 3 weeks of data to forecast
this week of data, or I can take the average of the last 3 weeks of data to forecast
next weeks data, leaving a gap of a window in between the window feature and the
forecasting point.

:class:`WindowFeatures` transformer works on top of `pandas.rolling`, `pandas.aggregate`
and `pandas.shift`. With `pandas.rolling`, :class:`WindowFeatures` determines the size
of the windows for the operations. With `pandas.rolling` we can specify the window size
with an integer, a string or a function. With :class:`WindowFeatures`, in addition, we
can pass a list of integers, strings or functions, to perform computations over multiple
window sizes.

:class:`WindowFeatures` uses `pandas.aggregate` to perform the mathematical operations
over the windows. Therefore, you can use any operation supported
by pandas. For supported aggregation functions, see Rolling Window
`Functions <https://pandas.pydata.org/docs/reference/window.html>`_.

With `pandas.shift`, :class:`WindowFeatures` places the value derived from the past
window, at the place of the value that we want to forecast. So if we want to forecast
this week with the average of the past 3 weeks of data, we should shift the value 1
week forward. If we want to forecast next week with the last 3 weeks of data, we should
forward the value 2 weeks forward.

:class:`WindowFeatures` will add the new variables with a representative name to the
original dataframe. It also has the methods `fit()` and `transform()` that make it
compatible with the Scikit-learn's `Pipeline` and cross-validation functions.

Note that to be compatible with :class:`WindowFeatures` the dataframe's index must have
unique values and no NaN.

Examples
--------

Let's create a toy dataset to demonstrate the functionality of :class:`WindowFeatures`.
The dataframe contains 3 numerical variables, a categorical variable, and a datetime
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



Now we will create window features from the numerical variables. By setting
`window=["30min", "60min"]` we perform calculations over windows of 30 and 60
minutes, respectively. If you look at our toy dataframe, you'll notice that 30 minutes
corresponds to 2 rows of data, and 60 minutes are 4 rows of data. So, we will perform calculations
over 2 and then 4 rows of data, respectively.

In `functions`, we indicate all the operations that we want to perform over those windows.
In our example below, we want to calculate the mean and the standard deviation of the
data within those windows and also find the maximum value within the windows.

With `freq="15min"` we indicate that we need to shift the calculations 15 minutes
forward. In other words, we are using the data available in windows defined up to 15 minutes
before the forecasting point.

.. code:: python

    from feature_engine.timeseries.forecasting import WindowFeatures

    win_f = WindowFeatures(
        window=["30min", "60min"], functions=["mean", "max", "std"], freq="15min",
    )

    X_tr = win_f.fit_transform(X)

    X_tr.head()

We can find the window features on the right side of the dataframe.

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                         ambient_temp_window_30min_mean  \
    2020-05-15 12:00:00                             NaN
    2020-05-15 12:15:00                           31.31
    2020-05-15 12:30:00                           31.41
    2020-05-15 12:45:00                           31.83
    2020-05-15 13:00:00                           32.27

                         ambient_temp_window_30min_max  \
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                          31.31
    2020-05-15 12:30:00                          31.51
    2020-05-15 12:45:00                          32.15
    2020-05-15 13:00:00                          32.39

                         ambient_temp_window_30min_std  \
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                            NaN
    2020-05-15 12:30:00                       0.141421
    2020-05-15 12:45:00                       0.452548
    2020-05-15 13:00:00                       0.169706

                         module_temp_window_30min_mean  \
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                         49.180
    2020-05-15 12:30:00                         49.510
    2020-05-15 12:45:00                         51.095
    2020-05-15 13:00:00                         51.490

                         module_temp_window_30min_max  \
    2020-05-15 12:00:00                           NaN
    2020-05-15 12:15:00                         49.18
    2020-05-15 12:30:00                         49.84
    2020-05-15 12:45:00                         52.35
    2020-05-15 13:00:00                         52.35

                         module_temp_window_30min_std  ...  \
    2020-05-15 12:00:00                           NaN  ...
    2020-05-15 12:15:00                           NaN  ...
    2020-05-15 12:30:00                      0.466690  ...
    2020-05-15 12:45:00                      1.774838  ...
    2020-05-15 13:00:00                      1.216224  ...

                         irradiation_window_30min_std  \
    2020-05-15 12:00:00                           NaN
    2020-05-15 12:15:00                           NaN
    2020-05-15 12:30:00                      0.197990
    2020-05-15 12:45:00                      0.098995
    2020-05-15 13:00:00                      0.077782

                         ambient_temp_window_60min_mean  \
    2020-05-15 12:00:00                             NaN
    2020-05-15 12:15:00                       31.310000
    2020-05-15 12:30:00                       31.410000
    2020-05-15 12:45:00                       31.656667
    2020-05-15 13:00:00                       31.840000

                         ambient_temp_window_60min_max  \
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                          31.31
    2020-05-15 12:30:00                          31.51
    2020-05-15 12:45:00                          32.15
    2020-05-15 13:00:00                          32.39

                         ambient_temp_window_60min_std  \
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                            NaN
    2020-05-15 12:30:00                       0.141421
    2020-05-15 12:45:00                       0.438786
    2020-05-15 13:00:00                       0.512640

                         module_temp_window_60min_mean  \
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                      49.180000
    2020-05-15 12:30:00                      49.510000
    2020-05-15 12:45:00                      50.456667
    2020-05-15 13:00:00                      50.500000

                         module_temp_window_60min_max  \
    2020-05-15 12:00:00                           NaN
    2020-05-15 12:15:00                         49.18
    2020-05-15 12:30:00                         49.84
    2020-05-15 12:45:00                         52.35
    2020-05-15 13:00:00                         52.35

                         module_temp_window_60min_std  \
    2020-05-15 12:00:00                           NaN
    2020-05-15 12:15:00                           NaN
    2020-05-15 12:30:00                      0.466690
    2020-05-15 12:45:00                      1.672553
    2020-05-15 13:00:00                      1.368381

                         irradiation_window_60min_mean  \
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                         0.5100
    2020-05-15 12:30:00                         0.6500
    2020-05-15 12:45:00                         0.6500
    2020-05-15 13:00:00                         0.6775

                         irradiation_window_60min_max  \
    2020-05-15 12:00:00                           NaN
    2020-05-15 12:15:00                          0.51
    2020-05-15 12:30:00                          0.79
    2020-05-15 12:45:00                          0.79
    2020-05-15 13:00:00                          0.79

                         irradiation_window_60min_std
    2020-05-15 12:00:00                           NaN
    2020-05-15 12:15:00                           NaN
    2020-05-15 12:30:00                      0.197990
    2020-05-15 12:45:00                      0.140000
    2020-05-15 13:00:00                      0.126853

    [5 rows x 22 columns]


The variables used as input for the window features are stored in the `variables_`
attribute of the :class:`WindowFeatures`:

.. code:: python

    win_f.variables_

.. code:: python

    ['ambient_temp', 'module_temp', 'irradiation']

We can obtain the names of the variables in the returned dataframe using the
get_feature_names_out() method:

.. code:: python

    win_f.get_feature_names_out()

.. code:: python

    ['ambient_temp',
     'module_temp',
     'irradiation',
     'color',
     'ambient_temp_window_30min_mean',
     'ambient_temp_window_30min_max',
     'ambient_temp_window_30min_std',
     'module_temp_window_30min_mean',
     'module_temp_window_30min_max',
     'module_temp_window_30min_std',
     'irradiation_window_30min_mean',
     'irradiation_window_30min_max',
     'irradiation_window_30min_std',
     'ambient_temp_window_60min_mean',
     'ambient_temp_window_60min_max',
     'ambient_temp_window_60min_std',
     'module_temp_window_60min_mean',
     'module_temp_window_60min_max',
     'module_temp_window_60min_std',
     'irradiation_window_60min_mean',
     'irradiation_window_60min_max',
     'irradiation_window_60min_std']


Working with pandas series
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your time series is a pandas Series instead of a pandas Dataframe, you need to
transform it into a dataframe before using :class:`WindowFeatures`.

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

We can use :class:`WindowFeatures` to create, for example, 2 new window features by finding
the mean and maximum value within a 45 minute windows of a pandas Series if we convert it
to a pandas Dataframe using the method `to_frame()`:

.. code:: python

    win_f = WindowFeatures(
        window=["45min"],
        functions=["mean", "max"],
        freq="30min",
    )

    X_tr = win_f.fit_transform(X['ambient_temp'].to_frame())

    X_tr.head()

.. code:: python

                         ambient_temp  ambient_temp_window_45min_mean  \
    2020-05-15 12:00:00         31.31                             NaN
    2020-05-15 12:15:00         31.51                             NaN
    2020-05-15 12:30:00         32.15                       31.310000
    2020-05-15 12:45:00         32.39                       31.410000
    2020-05-15 13:00:00         32.62                       31.656667

                         ambient_temp_window_45min_max
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                            NaN
    2020-05-15 12:30:00                          31.31
    2020-05-15 12:45:00                          31.51
    2020-05-15 13:00:00                          32.15


And if we do not want the original values of time series in the returned dataframe, we
just need to remember to drop the original series after the transformation:

.. code:: python

    win_f = WindowFeatures(
        window=["45min"],
        functions=["mean", "max"],
        freq="30min",
        drop_original=True,
    )

    X_tr = win_f.fit_transform(X['ambient_temp'].to_frame())

    X_tr.head()

.. code:: python

                         ambient_temp_window_45min_mean  \
    2020-05-15 12:00:00                             NaN
    2020-05-15 12:15:00                             NaN
    2020-05-15 12:30:00                       31.310000
    2020-05-15 12:45:00                       31.410000
    2020-05-15 13:00:00                       31.656667

                         ambient_temp_window_45min_max
    2020-05-15 12:00:00                            NaN
    2020-05-15 12:15:00                            NaN
    2020-05-15 12:30:00                          31.31
    2020-05-15 12:45:00                          31.51
    2020-05-15 13:00:00                          32.15



Getting the name of the variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can easily obtain the name of the original and new variables with the method
`get_feature_names_out`. By using the method with the default parameters, we obtain
all the features in the output dataframe.

.. code:: python

    win_f = WindowFeatures()

    win_f.fit(X)

    win_f.get_feature_names_out()

.. code:: python

    ['ambient_temp',
     'module_temp',
     'irradiation',
     'color',
     'ambient_temp_window_3_mean',
     'module_temp_window_3_mean',
     'irradiation_window_3_mean']

