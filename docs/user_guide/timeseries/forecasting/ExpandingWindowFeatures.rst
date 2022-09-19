.. _expanding_window_features:

.. currentmodule:: feature_engine.timeseries.forecasting

ExpandingWindowFeatures
=======================

:class:`ExpandingWindowFeatures` adds expanding window features to the dataframe.
Window features are the result of applying an aggregation operation (e.g., mean,
min, max, etc.) to a variable over a past window. For an expanding window feature,
the window spans from the start of the data up to, but not including, the
time point at which the feature is being computed.

For example, the mean value of all the data prior to the current value is an
expanding window feature. The maximum value of all the rows prior to the
current row is another expanding window feature.

When forecasting the future values of a variable, the past values of that variable are
likely to be predictive. To capitalize on the past values of a variable, we can simply
lag features with :class:`LagFeatures`. We can also create features that summarise all
the past values into a single quantity utilising :class:`ExpandingWindowFeatures`.

:class:`ExpandingWindowFeatures` works on top of `pandas.expanding`, `pandas.aggregate`
and `pandas.shift`.

:class:`ExpandingWindowFeatures` uses `pandas.aggregate` to perform the mathematical
operations over the expanding window. Therefore, you can use any operation supported
by pandas. For supported aggregation functions, see Expanding Window
`Functions <https://pandas.pydata.org/docs/reference/window.html>`_.

With `pandas.shift`, :class:`ExpandingWindowFeatures` lags the result of the expanding
window operation. This is useful to ensure that only the information known at predict
time is used to compute the window feature. So if at predict time we only know
the value of a feature at the previous time period and before that, then we should lag the
the window feature by 1 period. If at predict time we only know the value of a feature
from 2 weeks ago and before that, then we should lag the window feature column by 2 weeks.
:class:`ExpandingWindowFeatures` uses a default lag of one period.

:class:`ExpandingWindowFeatures` will add the new variables with a representative
name to the original dataframe. It also has the methods `fit()` and `transform()`
that make it compatible with the Scikit-learn's `Pipeline` and cross-validation
functions.

Note that to be compatible with :class:`ExpandingWindowFeatures` the dataframe's
index must have unique values and no NaN.

Examples
--------

Let's create a toy dataset to demonstrate the functionality of :class:`ExpandingWindowFeatures`.
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



Now we will create expanding window features from the numerical variables. In `functions`,
we indicate all the operations that we want to perform over those windows. In
our example below, we want to calculate the mean and the standard deviation of the
data within those windows and also find the maximum value within the windows.

.. code:: python

    from feature_engine.timeseries.forecasting import ExpandingWindowFeatures

    win_f = ExpandingWindowFeatures(functions=["mean", "max", "std"])

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

                         ambient_temp_expanding_mean  ambient_temp_expanding_max  \
    2020-05-15 12:00:00                          NaN                         NaN
    2020-05-15 12:15:00                    31.310000                       31.31
    2020-05-15 12:30:00                    31.410000                       31.51
    2020-05-15 12:45:00                    31.656667                       32.15
    2020-05-15 13:00:00                    31.840000                       32.39

                         ambient_temp_expanding_std  module_temp_expanding_mean  \
    2020-05-15 12:00:00                         NaN                         NaN
    2020-05-15 12:15:00                         NaN                   49.180000
    2020-05-15 12:30:00                    0.141421                   49.510000
    2020-05-15 12:45:00                    0.438786                   50.456667
    2020-05-15 13:00:00                    0.512640                   50.500000

                         module_temp_expanding_max  module_temp_expanding_std  \
    2020-05-15 12:00:00                        NaN                        NaN
    2020-05-15 12:15:00                      49.18                        NaN
    2020-05-15 12:30:00                      49.84                   0.466690
    2020-05-15 12:45:00                      52.35                   1.672553
    2020-05-15 13:00:00                      52.35                   1.368381

                         irradiation_expanding_mean  irradiation_expanding_max  \
    2020-05-15 12:00:00                         NaN                        NaN
    2020-05-15 12:15:00                      0.5100                       0.51
    2020-05-15 12:30:00                      0.6500                       0.79
    2020-05-15 12:45:00                      0.6500                       0.79
    2020-05-15 13:00:00                      0.6775                       0.79

                         irradiation_expanding_std
    2020-05-15 12:00:00                        NaN
    2020-05-15 12:15:00                        NaN
    2020-05-15 12:30:00                   0.197990
    2020-05-15 12:45:00                   0.140000
    2020-05-15 13:00:00                   0.126853


The variables used as input for the window features are stored in the `variables_`
attribute of the :class:`ExpandingWindowFeatures`.

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
     'ambient_temp_expanding_mean',
     'ambient_temp_expanding_max',
     'ambient_temp_expanding_std',
     'module_temp_expanding_mean',
     'module_temp_expanding_max',
     'module_temp_expanding_std',
     'irradiation_expanding_mean',
     'irradiation_expanding_max',
     'irradiation_expanding_std']


Working with pandas series
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your time series is a pandas Series instead of a pandas Dataframe, you need to
transform it into a dataframe before using :class:`ExpandingWindowFeatures`.

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

We can use :class:`ExpandingWindowFeatures` to create, for example, 2 new expanding window
features by finding the mean and maximum value of a pandas Series if we convert
it to a pandas Dataframe using the method `to_frame()`:

.. code:: python

    win_f = ExpandingWindowFeatures(functions=["mean", "max"])

    X_tr = win_f.fit_transform(X['ambient_temp'].to_frame())

    X_tr.head()

.. code:: python

                         ambient_temp  ambient_temp_expanding_mean  \
    2020-05-15 12:00:00         31.31                          NaN
    2020-05-15 12:15:00         31.51                    31.310000
    2020-05-15 12:30:00         32.15                    31.410000
    2020-05-15 12:45:00         32.39                    31.656667
    2020-05-15 13:00:00         32.62                    31.840000

                         ambient_temp_expanding_max
    2020-05-15 12:00:00                         NaN
    2020-05-15 12:15:00                       31.31
    2020-05-15 12:30:00                       31.51
    2020-05-15 12:45:00                       32.15
    2020-05-15 13:00:00                       32.39


And if we do not want the original values of time series in the returned dataframe, we
just need to remember to drop the original series after the transformation:

.. code:: python

    win_f = WindowFeatures(
        functions=["mean", "max"],
        drop_original=True,
    )

    X_tr = win_f.fit_transform(X['ambient_temp'].to_frame())

    X_tr.head()

.. code:: python

                         ambient_temp_expanding_mean  ambient_temp_expanding_max
    2020-05-15 12:00:00                          NaN                         NaN
    2020-05-15 12:15:00                    31.310000                       31.31
    2020-05-15 12:30:00                    31.410000                       31.51
    2020-05-15 12:45:00                    31.656667                       32.15
    2020-05-15 13:00:00                    31.840000                       32.39

Getting the name of the variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can easily obtain the name of the original and new variables with the method
`get_feature_names_out`. By using the method with the default parameters, we obtain
all the features in the output dataframe.

.. code:: python

    win_f = ExpandingWindowFeatures()

    win_f.fit(X)

    win_f.get_feature_names_out()

.. code:: python

    ['ambient_temp',
     'module_temp',
     'irradiation',
     'color',
     'ambient_temp_expanding_mean',
     'module_temp_expanding_mean',
     'irradiation_expanding_mean']

