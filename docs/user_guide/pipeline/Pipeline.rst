.. _pipeline:

.. currentmodule:: feature_engine.pipeline

Pipeline
========
:class:`Pipeline` facilitates the chaining together of multiple estimators into a unified sequence. This proves beneficial
as data processing frequently involves a predefined series of actions, such as feature selection, normalization, and
training a machine learning model.

Feature-engine's :class:`Pipeline` is different from scikit-learn's Pipeline in that our :class:`Pipeline` supports
transformers that remove rows from the dataset, like `DropMissingData`, `OutlierTrimmer`, `LagFeatures` and `WindowFeatures`.

When observations are removed from the training data set, :class:`Pipeline` invokes the method `transform_x_y`
available in these transformers, to adjust the target variable to the remaining rows.

The Pipeline serves various functions in this context:


**Simplicity and encapsulation:**

You need only call the `fit` and `predict` functions once on your data to fit an entire sequence of estimators.

**Hyperparameter Optimization:**

Grid search and random search can be performed over hyperparameters of all estimators in the pipeline simultaneously.

**Safety**

Using a pipeline prevent the leakage of statistics from test data into the trained model during cross-validation, by
ensuring that the same data is used to fit the transformers and predictors.

Pipeline functions
------------------

Calling the `fit` function on the pipeline, is the same as calling `fit` on each individual estimator sequentially,
transforming the input data and forwarding it to the subsequent step.

The pipeline will have all the methods present in the final estimator within it. For instance, if the last estimator is
a classifier, the Pipeline can function as a classifier. Similarly, if the last estimator is a transformer, the pipeline inherits this functionality as well.

Setting up a Pipeline
---------------------

The :class:`Pipeline` is constructed utilizing a list of (key, value) pairs, wherein the key represents the desired
name for the step, and the value denotes an estimator or a transformer object.

In the following example, we set up a :class:`Pipeline` that drops missing data, then replaces categories with ordinal
numbers, and finally fits a Lasso regression model.

.. code:: python

    import numpy as np
    import pandas as pd
    from feature_engine.imputation import DropMissingData
    from feature_engine.encoding import OrdinalEncoder
    from feature_engine.pipeline import Pipeline

    from sklearn.linear_model import Lasso

    X = pd.DataFrame(
        dict(
            x1=[2, 1, 1, 0, np.nan],
            x2=["a", np.nan, "b", np.nan, "a"],
        )
    )
    y = pd.Series([1, 2, 3, 4, 5])

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", OrdinalEncoder(encoding_method="arbitrary")),
            ("lasso", Lasso(random_state=10)),
        ]
    )
    # predict
    pipe.fit(X, y)
    preds_pipe = pipe.predict(X)
    preds_pipe

In the output we see the predictions made by the pipeline:

.. code:: python

    array([2., 2.])

Accessing Pipeline steps
------------------------

The :class:`Pipeline`'s estimators are stored as a list within the `steps` attribute. We can use slicing notation to
obtain a subset or partial pipeline within the Pipeline. This functionality is useful for executing specific
transformations or their inverses selectively.

For example, this notation extracts the first step of the pipeline:

.. code:: python

   pipe[:1]

.. code:: python

   Pipeline(steps=[('drop', DropMissingData())])


This notation extracts the first **two** steps of the pipeline:

.. code:: python

   pipe[:2]

.. code:: python

   Pipeline(steps=[('drop', DropMissingData()),
                ('enc', OrdinalEncoder(encoding_method='arbitrary'))])


This notation extracts the last step of the pipeline:

.. code:: python

   pipe[-1:]

.. code:: python

   Pipeline(steps=[('lasso', Lasso(random_state=10))])

We can also select specific steps of the pipeline to check their attributes. For example,
we can check the coefficients of the Lasso algorithm as follows:

.. code:: python

    pipe.named_steps["lasso"].coef_

And we see the coefficients:

.. code:: python

    array([-0.,  0.])

There was no relationship between the target and the variables, so it's fine to obtain
these coefficients.

Let's instead check the ordinal encoder mappings for the categorical variables:

.. code:: python

    pipe.named_steps["enc"].encoder_dict_

We see the integers used to replace each category:

.. code:: python

    {'x2': {'a': 0, 'b': 1}}

Finding feature names in a Pipeline
-----------------------------------

The :class:`Pipeline` includes a `get_feature_names_out()` method, similar to other transformers. By employing
pipeline slicing, you can obtain the feature names entering each step.

Let's set up a Pipeline that adds new features to the dataset to make this more interesting:

.. code:: python

    import numpy as np
    import pandas as pd
    from feature_engine.imputation import DropMissingData
    from feature_engine.encoding import OneHotEncoder
    from feature_engine.pipeline import Pipeline

    from sklearn.linear_model import Lasso

    X = pd.DataFrame(
        dict(
            x1=[2, 1, 1, 0, np.nan],
            x2=["a", np.nan, "b", np.nan, "a"],
        )
    )
    y = pd.Series([1, 2, 3, 4, 5])

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", OneHotEncoder()),
            ("lasso", Lasso(random_state=10)),
        ]
    )
    pipe.fit(X, y)

In the first step of the pipeline, no features are added, we just drop rows with `nan`. So if we execute
`get_feature_names_out()` we should see just the 2 variables from the input dataframe:

.. code:: python

    pipe[:1].get_feature_names_out()

.. code:: python

    ['x1', 'x2']

In the second step, we add binary variables for each category of x2, so x2 should disappear, and in its place, we
should see the binary variables:

.. code:: python

    pipe[:2].get_feature_names_out()

.. code:: python

    ['x1', 'x2_a', 'x2_b']

The last step is an estimator, that is, a machine learning model. Estimators don't support the method
`get_feature_names_out()`. So if we apply this method to the entire pipeline, we'll get an error.


Accessing nested parameters
---------------------------

We can re-define, or re-set the parameters of the transformers and estimators within the pipeline. This is done under
the hood by the Grid search and random search. But in case you need to change a parameter in a step of the
:class:`Pipeline`, this is how you do it:

.. code:: python

    pipe.set_params(lasso__alpha=10)

Here, we changed the alpha of the lasso regression algorithm to 10.


Best use: Dropping rows during data preprocessing
-------------------------------------------------

Feature-engine's :class:`Pipeline` was designed to support transformers that remove rows from the dataset, like
`DropMissingData`, `OutlierTrimmer`, `LagFeatures` and `WindowFeatures`.

We saw earlier in this page how to use :class:`Pipeline` with `DropMissingData`. Let's now take a look at how to
combine :class:`Pipeline` with `LagFeatures` and `WindowFeaures` to do multiple step forecasting.

We start by making imports:

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
    from feature_engine.pipeline import Pipeline

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

We'll predict the next 6 hours of energy demand. We'll use direct forecasting. Hence, we need to create 6 target
variables, one for each step in the horizon:

.. code:: python

    horizon = 6
    y = pd.DataFrame(index=df.index)
    for h in range(horizon):
        y[f"h_{h}"] = df.shift(periods=-h, freq="h")
    y.dropna(inplace=True)
    df = df.loc[y.index]
    print(y.head())

This is our target variable:

.. code:: python

                                 h_0          h_1          h_2          h_3  \
    date_time
    2002-01-01 00:00:00  6919.366092  7165.974188  6406.542994  5815.537828
    2002-01-01 01:00:00  7165.974188  6406.542994  5815.537828  5497.732922
    2002-01-01 02:00:00  6406.542994  5815.537828  5497.732922  5385.851060
    2002-01-01 03:00:00  5815.537828  5497.732922  5385.851060  5574.731890
    2002-01-01 04:00:00  5497.732922  5385.851060  5574.731890  5457.770634

                                 h_4          h_5
    date_time
    2002-01-01 00:00:00  5497.732922  5385.851060
    2002-01-01 01:00:00  5385.851060  5574.731890
    2002-01-01 02:00:00  5574.731890  5457.770634
    2002-01-01 03:00:00  5457.770634  5698.152000
    2002-01-01 04:00:00  5698.152000  5938.337614

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
        periods=[1, 2, 3, 4, 5, 6],
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

Now, we assemble the steps in the :class:`Pipeline` and fit it to the training data:

.. code:: python

    pipe = Pipeline(
        [
            ("lagf", lagf),
            ("winf", winf),
            ("lasso", lasso),
        ]
    ).set_output(transform="pandas")

    pipe.fit(X_train, y_train)

We can obtain the datasets with the predictors and the targets like this:

.. code:: python

    Xt, yt = pipe[:-1].transform_x_y(X_test, y_test)

    X_test.shape, y_test.shape, Xt.shape, yt.shape

We see that the :class:`Pipeline` has dropped some rows during the transformation and re-adjusted the target.
The rows that were dropped were those necessary to create the first lags.

.. code:: python

    ((1417, 1), (1417, 6), (1410, 7), (1410, 6))

We can examine the predictors training set, to make sure we are passing the right variables
to the regression model:

.. code:: python

    print(Xt.head())

We see the input features:

.. code:: python

                         demand_lag_1  demand_lag_2  demand_lag_3  demand_lag_4  \
    date_time
    2015-01-01 01:00:00   7804.086240   8352.992140   7571.301440   7516.472988
    2015-01-01 02:00:00   7174.339984   7804.086240   8352.992140   7571.301440
    2015-01-01 03:00:00   6654.283364   7174.339984   7804.086240   8352.992140
    2015-01-01 04:00:00   6429.598010   6654.283364   7174.339984   7804.086240
    2015-01-01 05:00:00   6412.785284   6429.598010   6654.283364   7174.339984

                         demand_lag_5  demand_lag_6  demand_window_3h_mean
    date_time
    2015-01-01 01:00:00   7801.201802   7818.461408            7804.086240
    2015-01-01 02:00:00   7516.472988   7801.201802            7489.213112
    2015-01-01 03:00:00   7571.301440   7516.472988            7210.903196
    2015-01-01 04:00:00   8352.992140   7571.301440            6752.740453
    2015-01-01 05:00:00   7804.086240   8352.992140            6498.888886

Now, we can make forecasts for the test set:

.. code:: python

    forecast = pipe.predict(X_test)

    forecasts = pd.DataFrame(
        pipe.predict(X_test),
        index=Xt.loc[end_train:].index,
        columns=[f"step_{i+1}" for i in range(6)]

    )

    print(forecasts.head())

We see the 6 hr ahead energy demand prediction for each hour:

.. code:: python

                             step_1       step_2       step_3       step_4  \
    date_time
    2015-01-01 01:00:00  7810.769000  7890.897914  8123.247406  8374.365708
    2015-01-01 02:00:00  7049.673468  7234.890108  7586.593627  7889.608312
    2015-01-01 03:00:00  6723.246357  7046.660134  7429.115933  7740.984091
    2015-01-01 04:00:00  6639.543752  6962.661308  7343.941881  7616.240318
    2015-01-01 05:00:00  6634.279747  6949.262247  7287.866893  7633.157948

                              step_5       step_6
    date_time
    2015-01-01 01:00:00  8569.220349  8738.027713
    2015-01-01 02:00:00  8116.631154  8270.579148
    2015-01-01 03:00:00  7937.918837  8170.531420
    2015-01-01 04:00:00  7884.815566  8197.598425
    2015-01-01 05:00:00  7979.920512  8321.363714

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

Hyperparameter optimization
---------------------------

We can optimize the hyperparameters of the transformers and the estimators from a pipeline simultaneously.

We'll start by loading the titanic dataset:

.. code:: python

    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import OneHotEncoder
    from feature_engine.outliers import OutlierTrimmer
    from feature_engine.pipeline import Pipeline

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler

    X, y = load_titanic(
        return_X_y_frame=True,
        predictors_only=True,
        handle_missing=True,
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )

    print(X_train.head())

We see the first 5 rows from the training set below:

.. code:: python

          pclass     sex        age  sibsp  parch     fare    cabin embarked
    501        2  female  13.000000      0      1  19.5000  Missing        S
    588        2  female   4.000000      1      1  23.0000  Missing        S
    402        2  female  30.000000      1      0  13.8583  Missing        C
    1193       3    male  29.881135      0      0   7.7250  Missing        Q
    686        3  female  22.000000      0      0   7.7250  Missing        Q

Now, we set up a Pipeline:

.. code:: python

    pipe = Pipeline(
        [
            ("outliers", OutlierTrimmer(variables=["age", "fare"])),
            ("enc", OneHotEncoder()),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(random_state=10)),
        ]
    )

We establish the hyperparameter space to search:

.. code:: python

    param_grid={
        'logit__C': [0.1, 10.],
        'enc__top_categories': [None, 5],
        'outliers__capping_method': ["mad", 'iqr']
    }

We do the grid search:

.. code:: python

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=2,
        refit=False,
    )

    grid.fit(X_train, y_train)

And we can see the best hyperparameters for each step:

.. code:: python

    grid.best_params_

.. code:: python

    {'enc__top_categories': None,
     'logit__C': 0.1,
     'outliers__capping_method': 'iqr'}

And the best accuracy obtained with these hyperparameters:

.. code:: python

    grid.best_score_

.. code:: python

    0.7843822843822843

Additional resources
--------------------

To learn more about feature engineering and data preprocessing, including missing data imputation, outlier removal or
capping, variable transformation and encoding, check out our online course and book:

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
