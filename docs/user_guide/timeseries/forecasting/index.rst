.. -*- mode: rst -*-
.. _forecasting:

.. currentmodule:: feature_engine.timeseries.forecasting

Forecasting Features
====================
Machine learning is becoming increasingly popular for time series forecasting because of its ability
to model complex relationships and handle large datasets. While traditional forecasting methods like
moving averages, ARIMA (autoregressive integrated moving averages), exponential smoothing, and others,
are effective in identifying trend and seasonality, machine learning algorithms like linear regression,
random forests, or gradient boosted machines, can model complex patterns and incorporate exogenous
features, thereby resulting in accurate predictions.

However, to use machine learning to forecast time series data, we need to extract relevant features
and convert the data into a tabular format, so that it can be framed as a regression problem.

Feature-engine's time series forecasting transformers give us the ability to extract and generate
useful features from time series to use for forecasting. They are built on top of Python libraries
such as Pandas and offer an interface to extract various features from temporal data simultaneously.

Time series forecasting involves learning from historical data observations to predict future values.
Feature-engine's offers various transformers for creating features from the past values.


Lag and Window Features
-----------------------

Trend and seasonality can be captured using lag and window features. In Feature-engine, we have
three transformers to extract these features.

Lag features
~~~~~~~~~~~~~

A lag feature at a given time step represents the value of the time series from a prior time step.
Feature engine's :class:`LagFeatures` implements lag features. These are straightforward to
compute and widely used in time series forecasting tasks. For instance, in sales forecasting,
you might include the sales from the previous day, week,
or even year to predict the sales for a given day. Lag features are also the foundation
to many autoregressive models like moving averages and ARIMA.

While lag features are particularly effective for capturing seasonality, they can also model
short-term trends. Seasonality is well captured by lag features because they carry data at
regular intervals from the past - like daily, weekly, or yearly cycles. For example, a 7-day
lag can capture weekly seasonality, such as sales spikes over the weekend; and a 365-day lag
can capture yearly seasonality, like Christmas holiday sales. Hence a machine learning model
can understand a lot of the time series patterns by using lag features as input.

However, lag features may not capture long-term trends unless they are combined with other features,
such as rolling window or expanding window features.

Window features
~~~~~~~~~~~~~~~

:class:`WindowFeatures`, also known as rolling window features, are used to summarize past behavior over a
fixed time period by computing statistics like mean, standard deviation, min, max, sum, etc. on the
time series variable. For instance, in sales forecasting, calculating the "mean" sales value of the
previous 4 weeks of data is a window feature.

Window features smooth out short-term fluctuations in the data allowing the model to capture trend.
This is somewhat similar to moving averages in traditional time series analysis.

In time series forecasting, we can aggregate data across multiple window features by applying
various mathematical operations across different sized time windows. Moreover, we can also combine
window features with lag features, for instance, by computing rolling statistics on lagged values,
which adds more depth to the feature set. This approach helps us generate a large number of features
and can capture both short-term and long-term patterns in historical data.

To determine which window sizes or lag combinations are useful, you can either perform time series
analyses to identify relevant window sizes, or you can use Feature-engine's
:ref:`feature selection transformers <selection_user_guide>`.
These are used to drop subsets of variables that are uninformative and have low predictive power
which in turn improves model performance. We can set aside validation data while training the
forecasting model to test different configurations and finally stick with the values that result
in minimal forecasting error.

Expanding Window Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`ExpandingWindowFeatures` are similar to rolling window features but instead
of a fixed-size window,
the window expands incrementally over time, starting from the beginning of the time series.
With each new time step, the window includes all prior observations, retaining a cumulative
summary of the data.

For example, "sum" of all sales values up to a given time step provides insights into the
ongoing trend i.e., whether the sales are increasing, decreasing or don't change with time.
This helps to capture long-term trends in the data and also model cumulative effects over time,
thus improving forecast accuracy.

Expanding window features are helpful in various data science use cases like demand forecasting
for supply chain optimization, stock price prediction etc.

Just like rolling window features, expanding window features can be used with various statistical
methods like mean, sum, standard deviation, min, max, among others. Unlike rolling window, we
don't need to specify a window size as the window is expanded automatically at each time step.


Datetime Features
-----------------

In addition to lag and window features, Feature-engine also offers transformers to extract
other attributes from the time series such as day_of_week, day_of_month, quarter, year, hour,
minute etc. directly from the datetime variable using :obj:`DatetimeFeatures <feature_engine.datetime.DatetimeFeatures>`.
These features are important to identify seasonal patterns, daily trends, especially when certain
time periods have strong correlation with the target variable.


Cyclical Features
------------------

In time series data, certain time-based attributes, such as month_of_year, day_of_week, etc. are
inherently cyclical. For example, after 12th month, the calendar resets to 1st month, and after
the 7th weekday, the calendar resets to 1st weekday. To inform the model of this periodic structure,
Feature-engine allows us to capture this behavior through the
:obj:`CyclicalFeatures <feature_engine.creation.CyclicalFeatures>` transformer.

CyclicalFeatures represents datetime variables using the sine and cosine transformation,
allowing the model to understand the continuity of time-based cycles. This approach overcomes
the limitations of ordinal or label encoding, where the end and beginning of a cycle
(e.g., 12th month to 1st month) would appear distant in the feature space while in reality
they are closer. These features can further enhance model's ability to capture seasonal and
repeating patterns in the timeseries.

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

Forecasting Features Transformers
---------------------------------

.. toctree::
   :maxdepth: 1

   LagFeatures
   WindowFeatures
   ExpandingWindowFeatures

|
|
|
