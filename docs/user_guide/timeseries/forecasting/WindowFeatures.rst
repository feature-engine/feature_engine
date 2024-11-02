.. _window_features:

.. currentmodule:: feature_engine.timeseries.forecasting

WindowFeatures
==============

Window features are commonly used in data science to forecast time series with traditional
machine learning models, like linear regression or gradient boosting machines. Window features
are created by performing mathematical operations over windows of past data.

For example, the mean “sales” value of the previous 3 months of data is a window feature.
The maximum “revenue” of the previous three rows of data is another window feature.

In time series forecasting, we want to predict future values of the time series. To do this,
we can create window features by performing mathematical operations over windows of past
values of the time series data. Then, we would use this features to predict the time series
with any regression model.


Rolling window features with pandas
-----------------------------------

Window features are the result of window operations over the variables. Rolling window operations are
operations that perform an aggregation over a **sliding partition** of past values of the time
series data.

A window feature is, then, a feature created after computing mathematical
functions (e.g., mean, min, max, etc.) within a window over the past data.

In Python, we can create window features by utilizing pandas method `rolling`. For example,
by executing:

.. code:: python

    X[["var_1", "var_2"].rolling(window=3).agg(["max", "mean"])

With the previous command, we create 2 window features for each variable, `var_1` and
`var_2`, by taking the maximum and average value of the current and 2 previous rows of data.

If we want to use those features for forecasting using traditional machine learning
algorithms, we also need to shift the window forward with pandas method `shift`:

.. code:: python

    X[["var_1", "var_2"].rolling(window=3).agg(["max", "mean"]).shift(period=1)

Shifting is important to ensure that we are using values strictly in the past, respect
to the point that we want to forecast.

Sliding window features with Feature-engine
-------------------------------------------

:class:`WindowFeatures` can automatically create and add window features to the dataframe, by performing
multiple mathematical operations over different window sizes over various numerical variables.

Thus, :class:`WindowFeatures` creates and adds new features to the data set automatically
through the use of windows over historical data.

Window features: parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create window features we need to determine a number of parameters. First, we need to
identify the size of the window or windows in which we will perform the operations. For
example, we can take the average of the variable over 3 months, or 6 weeks.

We also need to determine how far back is the window located respect to the data point we
want to forecast. For example, I can take the average of the last 3 weeks of data to forecast
this week of data, or I can take the average of the last 3 weeks of data to forecast next
weeks data, leaving a gap of a window in between the window feature and the forecasting point.

WindowFeatures: under the hood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`WindowFeatures` works on top of `pandas.rolling`, `pandas.aggregate` and
`pandas.shift`. With `pandas.rolling`, :class:`WindowFeatures` determines the size
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

:class:`WindowFeatures` will add the new features with a representative name to the
original dataframe. It also has the methods `fit()` and `transform()` that make it
compatible with the Scikit-learn's `Pipeline` and cross-validation functions.

Note that, in the current implementation, :class:`WindowFeatures` only works with dataframes whose index,
containing the time series timestamp, contains unique values and no NaN.

Examples
--------

Let's create a time series dataset to see how to create window features with
:class:`WindowFeatures`. The dataframe contains 3 numerical variables, a categorical
variable, and a datetime index. We also create a target variable.

.. code:: python

    import pandas as pd

    X = {"ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68],
         "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52],
         "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56],
         "color": ["green"] * 4 + ["blue"] * 4,
         }

    X = pd.DataFrame(X)
    X.index = pd.date_range("2020-05-15 12:00:00", periods=8, freq="15min")

    y = pd.Series([1,2,3,4,5,6,7,8])
    y.index = X.index

    X.head()

Below we see the dataframe:

.. code:: python

                         ambient_temp  module_temp  irradiation  color
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue


Let's now print out the target:

.. code:: python

    y

Below we see the target variable:

.. code:: python

    2020-05-15 12:00:00    1
    2020-05-15 12:15:00    2
    2020-05-15 12:30:00    3
    2020-05-15 12:45:00    4
    2020-05-15 13:00:00    5
    2020-05-15 13:15:00    6
    2020-05-15 13:30:00    7
    2020-05-15 13:45:00    8
    Freq: 15min, dtype: int64

Now we will create window features from the numerical variables. By setting
`window=["30min", "60min"]` we perform calculations over windows of 30 and 60
minutes, respectively.

If you look at our toy dataframe, you'll notice that 30 minutes corresponds to 2 rows of
data, and 60 minutes are 4 rows of data. So, we will perform calculations over 2 and then
4 rows of data, respectively.

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

We find the window features on the right side of the dataframe.

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

We can obtain the names of the variables in the transformed dataframe using the
`get_feature_names_out()` method:

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

Dropping rows with nan
~~~~~~~~~~~~~~~~~~~~~~

When we create window features, we may introduce nan values for those data points where
there isn't enough data in the past to create the windows. We can automatically drop
the rows with nan values in the window features both in the train set and in the target
variable as follows:

.. code:: python

    win_f = WindowFeatures(
        window=["30min", "60min"],
        functions=["mean", ],
        freq="15min",
        drop_na=True,
    )

    win_f.fit(X)

    X_tr, y_tr = win_f.transform_x_y(X, y)

    X.shape, y.shape, X_tr.shape, y_tr.shape

We see that the resulting dataframe contains less rows than the original dataframe:

.. code:: python

    ((8, 4), (8,), (7, 10), (7,))


Imputing rows with nan
~~~~~~~~~~~~~~~~~~~~~~

If instead of removing the row with nan in the window features, we want to impute those
values, we can do so with any of Feature-engine's imputers. Here, we will replace nan with
the arbitrary value -99, using the `ArbitraryNumberImputer` within a pipeline:


.. code:: python

    from feature_engine.imputation import ArbitraryNumberImputer
    from feature_engine.pipeline import Pipeline

    win_f = WindowFeatures(
        window=["30min", "60min"],
        functions=["mean", ],
        freq="15min",
    )

    pipe = Pipeline([
        ("windows", win_f),
        ("imputer", ArbitraryNumberImputer(arbitrary_number=-99))
    ])

    X_tr = pipe.fit_transform(X, y)

    print(X_tr.head())

We see the resulting dataframe, where the nan values were replaced by -99:

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                         ambient_temp_window_30min_mean  \
    2020-05-15 12:00:00                          -99.00
    2020-05-15 12:15:00                           31.31
    2020-05-15 12:30:00                           31.41
    2020-05-15 12:45:00                           31.83
    2020-05-15 13:00:00                           32.27

                         module_temp_window_30min_mean  \
    2020-05-15 12:00:00                        -99.000
    2020-05-15 12:15:00                         49.180
    2020-05-15 12:30:00                         49.510
    2020-05-15 12:45:00                         51.095
    2020-05-15 13:00:00                         51.490

                         irradiation_window_30min_mean  \
    2020-05-15 12:00:00                        -99.000
    2020-05-15 12:15:00                          0.510
    2020-05-15 12:30:00                          0.650
    2020-05-15 12:45:00                          0.720
    2020-05-15 13:00:00                          0.705

                         ambient_temp_window_60min_mean  \
    2020-05-15 12:00:00                      -99.000000
    2020-05-15 12:15:00                       31.310000
    2020-05-15 12:30:00                       31.410000
    2020-05-15 12:45:00                       31.656667
    2020-05-15 13:00:00                       31.840000

                         module_temp_window_60min_mean  \
    2020-05-15 12:00:00                     -99.000000
    2020-05-15 12:15:00                      49.180000
    2020-05-15 12:30:00                      49.510000
    2020-05-15 12:45:00                      50.456667
    2020-05-15 13:00:00                      50.500000

                         irradiation_window_60min_mean
    2020-05-15 12:00:00                       -99.0000
    2020-05-15 12:15:00                         0.5100
    2020-05-15 12:30:00                         0.6500
    2020-05-15 12:45:00                         0.6500
    2020-05-15 13:00:00                         0.6775

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



Getting the name of the new features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Windows from the target vs windows from predictor variables
-----------------------------------------------------------

Very often, we work with univariate time series, for example, the total sales revenue of a
retail company. We want to forecast future sales values. The sales variable is our target
variable, and we can extract features from windows of past sales values.

We could also work with multivariate time series, where we have sales in different
countries, or alternatively, multiple time series, like pollutant concentration in the
air, accompanied by concentrations of other gases, temperature, and humidity.

When working with multivariate time series, like sales in multiple countries, we would
extract features from windows of past data for each country separately.

When working with multiple time series, like pollutant concentration, gas concentration,
temperature, and humidity, pollutant concentration is our target variable, and the other
time series are accompanying predictive variables. We can create window features from
past pollutant concentrations, that is, from past time steps of our target variable.
And, in addition, we can create features from windows of past data from accompanying
time series, like the concentrations of other gases or the temperature or humidity.

The process of feature extraction from time series data, to create a table of predictors
and a target variable to forecast using supervised learning models like linear regression
or random forest, is called “tabularizing” the time series.

See also
--------

Check out the additional transformers to create expanding window features
(:class:`ExpandingWindowFeatures`) or lag features, by lagging past values of the time
series data (:class:`LagFeatures`).

Other open-source packages for window features
----------------------------------------------

- `tsfresh <https://tsfresh.readthedocs.io/en/latest/text/forecasting.html>`_
- `featuretools <https://featuretools.alteryx.com/en/stable/guides/time_series.html>`_

Tutorials and courses
---------------------

For tutorials about this and other feature engineering methods for time series forecasting
check out our online courses:

.. figure::  ../../../images/fetsf.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-forecasting

   Feature Engineering for Time Series Forecasting

.. figure::  ../../../images/fwml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.courses.trainindata.com/p/forecasting-with-machine-learning

   Forecasting with Machine Learning

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
|
|
|
|
|
|
|


Our courses are suitable for beginners and more advanced data scientists looking to
forecast time series using traditional machine learning models, like linear regression
or gradient boosting machines.

By purchasing them you are supporting Sole, the main developer of Feature-engine.