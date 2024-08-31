.. _make_pipeline:

.. currentmodule:: feature_engine.pipeline

make_pipeline
=============

:class:`make_pipeline` is a shorthand for :class:`Pipeline`. While to set up :class:`Pipeline`
we create tuples with step names and transformers or estimators, with :class:`make_pipeline`
we just add a sequence of transformers and estimators, and the names will be added automatically.

Setting up a Pipeline with make_pipeline
----------------------------------------

In the following example, we set up a `Pipeline` that drops missing data, then replaces categories with ordinal
numbers, and finally fits a Lasso regression model.

.. code:: python

    import numpy as np
    import pandas as pd
    from feature_engine.imputation import DropMissingData
    from feature_engine.encoding import OrdinalEncoder
    from feature_engine.pipeline import make_pipeline

    from sklearn.linear_model import Lasso

    X = pd.DataFrame(
        dict(
            x1=[2, 1, 1, 0, np.nan],
            x2=["a", np.nan, "b", np.nan, "a"],
        )
    )
    y = pd.Series([1, 2, 3, 4, 5])

    pipe = make_pipeline(
        DropMissingData(),
        OrdinalEncoder(encoding_method="arbitrary"),
        Lasso(random_state=10),
    )
    # predict
    pipe.fit(X, y)
    preds_pipe = pipe.predict(X)
    preds_pipe

In the output we see the predictions made by the pipeline:

.. code:: python

    array([2., 2.])

The names of the pipeline were assigned automatically:

.. code:: python

   print(pipe)

.. code:: python

    Pipeline(steps=[('dropmissingdata', DropMissingData()),
                    ('ordinalencoder', OrdinalEncoder(encoding_method='arbitrary')),
                    ('lasso', Lasso(random_state=10))])

The pipeline returned by :class:`make_pipeline` has exactly the same characteristics than
:class:`Pipeline`. Hence, for additional guidelines, check out the :class:`Pipeline`
documentation.

Forecasting
-----------

Let's set up another pipeline to do direct forecasting:

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    from sklearn.linear_model import Lasso
    from sklearn.metrics import root_mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor

    from feature_engine.timeseries.forecasting import (
        LagFeatures,
        WindowFeatures,
    )
    from feature_engine.pipeline import make_pipeline

We'll use the Australia electricity demand dataset described here:

Godahewa, Rakshitha, Bergmeir, Christoph, Webb, Geoff, Hyndman, Rob, & Montero-Manso, Pablo. (2021). Australian
Electricity Demand Dataset (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4659727

.. code:: python

    url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
    df = pd.read_csv(url)

    df.drop(columns=["Industrial"], inplace=True)

    # Convert the integer Date to an actual date with datetime type
    df["date"] = df["Date"].apply(
        lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
    )

    # Create a timestamp from the integer Period representing 30 minute intervals
    df["date_time"] = df["date"] + \
        pd.to_timedelta((df["Period"] - 1) * 30, unit="m")

    df.dropna(inplace=True)

    # Rename columns
    df = df[["date_time", "OperationalLessIndustrial"]]

    df.columns = ["date_time", "demand"]

    # Resample to hourly
    df = (
        df.set_index("date_time")
        .resample("h")
        .agg({"demand": "sum"})
    )

    print(df.head())

Here, we see the first rows of data:

.. code:: python

                              demand
    date_time
    2002-01-01 00:00:00  6919.366092
    2002-01-01 01:00:00  7165.974188
    2002-01-01 02:00:00  6406.542994
    2002-01-01 03:00:00  5815.537828
    2002-01-01 04:00:00  5497.732922

We'll predict the next 3 hours of energy demand. We'll use direct forecasting. Let's
create the target variable:

.. code:: python

    horizon = 3
    y = pd.DataFrame(index=df.index)
    for h in range(horizon):
        y[f"h_{h}"] = df.shift(periods=-h, freq="h")
    y.dropna(inplace=True)
    df = df.loc[y.index]
    print(y.head())

This is our target variable:

.. code:: python

                                 h_0          h_1          h_2
    date_time
    2002-01-01 00:00:00  6919.366092  7165.974188  6406.542994
    2002-01-01 01:00:00  7165.974188  6406.542994  5815.537828
    2002-01-01 02:00:00  6406.542994  5815.537828  5497.732922
    2002-01-01 03:00:00  5815.537828  5497.732922  5385.851060
    2002-01-01 04:00:00  5497.732922  5385.851060  5574.731890

Next, we split the data into a training set and a test set:

.. code:: python

    end_train = '2014-12-31 23:59:59'
    X_train = df.loc[:end_train]
    y_train = y.loc[:end_train]

    begin_test = '2014-12-31 17:59:59'
    X_test  = df.loc[begin_test:]
    y_test = y.loc[begin_test:]

Next, we set up `LagFeatures` and `WindowFeatures` to create features from lags and windows:

.. code:: python

    lagf = LagFeatures(
        variables=["demand"],
        periods=[1, 3, 6],
        missing_values="ignore",
        drop_na=True,
    )


    winf = WindowFeatures(
        variables=["demand"],
        window=["3h"],
        freq="1h",
        functions=["mean"],
        missing_values="ignore",
        drop_original=True,
        drop_na=True,
    )

We wrap the lasso regression within the multioutput regressor to predict multiple targets:

.. code:: python

    lasso = MultiOutputRegressor(Lasso(random_state=0, max_iter=10))

Now, we assemble `Pipeline`:

.. code:: python

    pipe = make_pipeline(lagf, winf, lasso)

    print(pipe)

The steps' names were assigned automatically:

.. code:: python

    Pipeline(steps=[('lagfeatures',
                     LagFeatures(drop_na=True, missing_values='ignore',
                                 periods=[1, 3, 6], variables=['demand'])),
                    ('windowfeatures',
                     WindowFeatures(drop_na=True, drop_original=True, freq='1h',
                                    functions=['mean'], missing_values='ignore',
                                    variables=['demand'], window=['3h'])),
                    ('multioutputregressor',
                     MultiOutputRegressor(estimator=Lasso(max_iter=10,
                                                          random_state=0)))])

Let's fit the Pipeline:

.. code:: python

    pipe.fit(X_train, y_train)

Now, we can make forecasts for the test set:

.. code:: python

    forecast = pipe.predict(X_test)

    forecasts = pd.DataFrame(
        pipe.predict(X_test),
        columns=[f"step_{i+1}" for i in range(3)]

    )

    print(forecasts.head())

We see the 3 hr ahead energy demand prediction for each hour:

.. code:: python

            step_1       step_2       step_3
    0  8031.043352  8262.804811  8484.551733
    1  7017.158081  7160.568853  7496.282999
    2  6587.938171  6806.903940  7212.741943
    3  6503.807479  6789.946587  7195.796841
    4  6646.981390  6970.501840  7308.359237


To learn more about direct forecasting and how to create features, check out our courses:


.. figure::  ../../images/fetsf.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-forecasting

   Feature Engineering for Time Series Forecasting

.. figure::  ../../images/fwml.png
   :width: 300
   :figclass: align-center
   :align: right
   :target: https://www.courses.trainindata.com/p/forecasting-with-machine-learning

   Forecasting with Machine Learning

.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

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