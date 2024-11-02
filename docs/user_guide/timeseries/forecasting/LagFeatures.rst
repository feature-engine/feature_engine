.. _lag_features:

.. currentmodule:: feature_engine.timeseries.forecasting

LagFeatures
===========

Lag features are commonly used in data science to forecast time series with traditional

machine learning models, like linear regression or random forests. A lag feature is a
feature with information about a prior time step of the time series.

When forecasting the future values of a variable, the past values of that same variable
are likely to be predictive. Past values of other predictive features can also be useful
for our forecast. Thus, in forecasting, it is common practice to create lag features from
time series data and use them as input to machine learning algorithms or forecasting
models.

What is a lag feature?
----------------------

A lag feature is the value of the time series **k** period(s) in the past, where **k** is
the lag and is to be set by the user.  For example, a lag of 1 is a feature that contains
the previous time point value of the time series. A lag of 3 contains the value 3 time
points before, and so on. By varying k, we can create features with multiple lags.

In Python, we can create lag features by using the pandas method `shift`. For example, by
executing `X[my_variable].shift(freq=”1H”, axis=0)`, we create a new feature consisting of
lagged values of `my_variable` by 1 hour.

Feature-engine’s :class:`LagFeatures` automates the creation of lag features from multiple
variables and by using multiple lags. It uses pandas `shift` under the hood, and automatically
concatenates the new features to the input dataframe.

Automating lag feature creation
-------------------------------

There are 2 ways in which we can indicate the lag k using :class:`LagFeatures`. Just like
with pandas `shift`, we can indicate the lag using the parameter `periods`. This parameter
takes integers that indicate the number of rows forward that the features will be lagged.

Alternatively, we can use the parameter `freq`, which takes a string with the period and
frequency, and lags features based on the datetime index. For example, if we pass `freq="1D"`,
the values of the features will be moved 1 day forward.

The :class:`LagFeatures` transformer works very similarly to `pandas.shift`, but unlike
`pandas.shift` we can indicate the lag using either `periods` or `freq` but not both at the
same time. Also, unlike `pandas.shift`, we can only lag features forward.

:class:`LagFeatures` has several advantages over `pandas.shift`:

- First, it can create features with multiple values of k at the same time.
- Second, it adds the features with a name to the original dataframe.
- Third, it has the methods `fit()` and `transform()` that make it compatible with the Scikit-learn's `Pipeline` and cross-validation functions.

Note that, in the current implementation, :class:`LagFeatures` only works with dataframes whose index,
containing the time series timestamp, contains unique values and no NaN.

Examples
--------

Let's create a toy dataset to show how to add lag features with :class:`LagFeatures`.
The dataframe contains 3 numerical variables, a categorical variable, and a datetime
index. We also create an arbitrary target.

.. code:: python

    import pandas as pd

    X = {"ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68],
         "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52],
         "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56],
         "color": ["green"] * 4 + ["blue"] * 4,
         }

    X = pd.DataFrame(X)
    X.index = pd.date_range("2020-05-15 12:00:00", periods=8, freq="15min")
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    y.index = X.index

    X.head()

Below we see the output of our toy dataframe:

.. code:: python

                         ambient_temp  module_temp  irradiation  color
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue


And here we print and show the target variable:

.. code:: python

    y

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

Shift a row forward
~~~~~~~~~~~~~~~~~~~

Now we will create lag features by lagging all numerical variables 1 row forward. Note
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

We can obtain the names of the original variables plus the lag features that are the
returned in the transformed dataframe using the `get_feature_names_out()` method:

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

When we create lag features, we introduce nan values for the first rows of the training
data set, because there are no past values for those data points. We can impute those
nan values with an arbitrary value as follows:

.. code:: python

    lag_f = LagFeatures(periods=1, fill_value=0)

    X_tr = lag_f.fit_transform(X)

    print(X_tr.head())

We see that the nan values were replaced by 0:

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                         ambient_temp_lag_1  module_temp_lag_1  irradiation_lag_1
    2020-05-15 12:00:00                0.00               0.00               0.00
    2020-05-15 12:15:00               31.31              49.18               0.51
    2020-05-15 12:30:00               31.51              49.84               0.79
    2020-05-15 12:45:00               32.15              52.35               0.65
    2020-05-15 13:00:00               32.39              50.63               0.76

Alternatively, we can drop the rows with missing values in the lag features, like this:

.. code:: python

    lag_f = LagFeatures(periods=1, drop_na=True)

    X_tr = lag_f.fit_transform(X)

    print(X_tr.head())

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue
    2020-05-15 13:15:00         32.50        47.01         0.49   blue

                         ambient_temp_lag_1  module_temp_lag_1  irradiation_lag_1
    2020-05-15 12:15:00               31.31              49.18               0.51
    2020-05-15 12:30:00               31.51              49.84               0.79
    2020-05-15 12:45:00               32.15              52.35               0.65
    2020-05-15 13:00:00               32.39              50.63               0.76
    2020-05-15 13:15:00               32.62              49.61               0.42

We can also drop the rows with nan in the lag features and then adjust the target
variable like this:

.. code:: python

    X_tr, y_tr = lag_f.transform_x_y(X, y)

    X_tr.shape, y_tr.shape, X.shape, y.shape

We created a lag feature of 1, hence there is only 1 row with nan, which was removed from
train set and target:

.. code:: python

    ((7, 7), (7,), (8, 4), (8,))

Create multiple lag features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can create multiple lag features with one transformer by passing the lag periods in
a list.

.. code:: python

    lag_f = LagFeatures(periods=[1, 2])

    X_tr = lag_f.fit_transform(X)

    X_tr.head()

Note how multiple lag features were created for each of the numerical variables and
added at the right side of the dataframe.

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

We can get the names of features in the resulting dataframe as follows:

.. code:: python

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

We can replace the nan introduced in the lag features as well. In this opportunity,
we'll use a string. Not that this is a suitable solution to train machine learning
algorithms, but the idea here is to showcase :class:`LagFeatures`'s functionality.

.. code:: python

    lag_f = LagFeatures(periods=[1, 2], fill_value='None')

    X_tr = lag_f.fit_transform(X)

    print(X_tr.head())

In this case, we replaced the nan in the lag features with the string None:

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                        ambient_temp_lag_1 module_temp_lag_1 irradiation_lag_1  \
    2020-05-15 12:00:00               None              None              None
    2020-05-15 12:15:00              31.31             49.18              0.51
    2020-05-15 12:30:00              31.51             49.84              0.79
    2020-05-15 12:45:00              32.15             52.35              0.65
    2020-05-15 13:00:00              32.39             50.63              0.76

                        ambient_temp_lag_2 module_temp_lag_2 irradiation_lag_2
    2020-05-15 12:00:00               None              None              None
    2020-05-15 12:15:00               None              None              None
    2020-05-15 12:30:00              31.31             49.18              0.51
    2020-05-15 12:45:00              31.51             49.84              0.79
    2020-05-15 13:00:00              32.15             52.35              0.65

Alternatively, we can drop rows containing nan in the lag features and then adjust the
target variable:

.. code:: python

    lag_f = LagFeatures(periods=[1, 2], drop_na=True)

    lag_f.fit(X)

    X_tr, y_tr = lag_f.transform_x_y(X, y)

    X_tr.shape, y_tr.shape, X.shape, y.shape

We see that 2 rows were dropped from train set and target:

.. code:: python

    ((6, 10), (6,), (8, 4), (8,))

Lag features based on datetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also lag features utilizing information in the timestamp of the dataframe, which
is commonly cast as datetime.

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

We can replace the nan in the lag features with a number like this:

.. code:: python

    lag_f = LagFeatures(
        variables=["module_temp", "irradiation"], freq="30min", fill_value=100)

    X_tr = lag_f.fit_transform(X)

    print(X_tr.head())

Here, we replaced nan by 100:

.. code:: python

                         ambient_temp  module_temp  irradiation  color  \
    2020-05-15 12:00:00         31.31        49.18         0.51  green
    2020-05-15 12:15:00         31.51        49.84         0.79  green
    2020-05-15 12:30:00         32.15        52.35         0.65  green
    2020-05-15 12:45:00         32.39        50.63         0.76  green
    2020-05-15 13:00:00         32.62        49.61         0.42   blue

                         module_temp_lag_30min  irradiation_lag_30min
    2020-05-15 12:00:00                 100.00                 100.00
    2020-05-15 12:15:00                 100.00                 100.00
    2020-05-15 12:30:00                  49.18                   0.51
    2020-05-15 12:45:00                  49.84                   0.79
    2020-05-15 13:00:00                  52.35                   0.65


Alternatively, we can remove the nan introduced in the lag features and adjust the target:

.. code:: python

    lag_f = LagFeatures(
        variables=["module_temp", "irradiation"], freq="30min", drop_na=True)

    lag_f.fit(X)

    X_tr, y_tr = lag_f.transform_x_y(X, y)

    X_tr.shape, y_tr.shape, X.shape, y.shape

Two rows were removed from the training data set and the target:

.. code:: python

    ((6, 6), (6,), (8, 4), (8,))

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

This is super useful in time series forecasting, because the original variable is usually
the one that we are trying to forecast, that is, the target variable. The original variables
also contain values that are **NOT** available at the time points that we are forecasting.

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

We can use :class:`LagFeatures` to create, for example, 3 features by lagging the
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


Getting the name of the lag features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Determining the right lag
-------------------------

We can create multiple lag features by utilizing various lags. But how do we decide which
lag is a good lag?

There are multiple ways to do this.

We can create features by using multiple lags and then determine the best features by using
feature selection.

Alternatively, we can determine the best lag through time series analysis by evaluating
the autocorrelation or partial autocorrelation of the time series.

For tutorials on how to create lag features for forecasting, check the course
`Feature Engineering for Time Series Forecasting <https://www.trainindata.com/p/feature-engineering-for-forecasting>`_.
In the course, we also show how to detect and remove outliers from time series data, how
to use features that capture seasonality and trend, and much more.

Lags from the target vs lags from predictor variables
-----------------------------------------------------
Very often, we want to forecast the values of just one time series. For example, we want
to forecast sales in the next month. The sales variable is our target variable, and we can
create features by lagging past sales values.

We could also create lag features from accompanying predictive variables. For example, if we
want to predict pollutant concentration in the next few hours, we can create lag features
from past pollutant concentrations. In addition, we can create lag features from accompanying
time series values, like the concentrations of other gases, or the temperature or humidity.

See also
--------

Check out the additional transformers to create window features through the use of
rolling windows (:class:`WindowFeatures`) or expanding windows (:class:`ExpandingWindowFeatures`).

If you want to use :class:`LagFeatures` as part of a feature engineering pipeline,
check out Feature-engine's `Pipeline`.

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