.. _drop_missing_data:

.. currentmodule:: feature_engine.imputation

DropMissingData
===============

Removing rows with nan values from a dataset is a common practice in data science and
machine learning projects.

You are probably familiar with the use of pandas dropna. You basically take a pandas
dataframe or a pandas series, apply dropna, and eliminate those rows that contain nan
values in one or more columns.

Here, we have an example of that syntax:

.. code:: python

    import numpy as np
    import pandas as pd

    X = pd.DataFrame(dict(
           x1 = [np.nan,1,1,0,np.nan],
           x2 = ["a", np.nan, "b", np.nan, "a"],
           ))

    X.dropna(inplace=True)
    print(X)

The previous code returns a dataframe without missing values:

.. code:: python

        x1 x2
    2  1.0  b

Feature-engine's :class:`DropMissingData()` wraps pandas dropna in a transformer that
will remove rows with na values while adhering to scikit-learn's `fit` and `transform`
functionality.

Here we have a snapshot of :class:`DropMissingData()`'s syntax:

.. code:: python

    import pandas as pd
    import numpy as np
    from feature_engine.imputation import DropMissingData

    X = pd.DataFrame(dict(
           x1 = [np.nan,1,1,0,np.nan],
           x2 = ["a", np.nan, "b", np.nan, "a"],
           ))

    dmd = DropMissingData()
    dmd.fit(X)
    dmd.transform(X)

The previous code returns a dataframe without missing values:

.. code:: python

        x1 x2
    2  1.0  b

:class:`DropMissingData()` allows you therefore to remove null values as part of any
scikit-learn feature engineering workflow.

DropMissingData
---------------

:class:`DropMissingData()` has some advantages over pandas:

- It learns and stores the variables for which rows with nan values should be deleted.
- It can be used within a Scikit-learn like pipeline.

With :class:`DropMissingData()`, you can drop nan values from numerical and categorical
variables. In other words, you can remove null values from numerical, categorical or
object datatypes.

You have the option to remove nan values from all columns or only from a subset of
them. Alternatively, you can remove rows if they have more than a certain percentage of
nan values.

Let's better illustrate :class:`DropMissingData()`'s functionality through code examples.

Dropna
^^^^^^

Let's start by importing pandas and numpy, and creating a toy dataframe with nan values
in 2 columns:

.. code:: python

    import numpy as np
    import pandas as pd

    from feature_engine.imputation import DropMissingData

    X = pd.DataFrame(
        dict(
            x1=[2, 1, 1, 0, np.nan],
            x2=["a", np.nan, "b", np.nan, "a"],
            x3=[2, 3, 4, 5, 5],
        )
    )
    y = pd.Series([1, 2, 3, 4, 5])

    print(X.head())

Below we see the new dataframe:

.. code:: python

        x1   x2  x3
    0  2.0    a   2
    1  1.0  NaN   3
    2  1.0    b   4
    3  0.0  NaN   5
    4  NaN    a   5

We can drop nan values across all columns as follows:

.. code:: python

    dmd =  DropMissingData()
    Xt = dmd.fit_transform(X)
    Xt.head()

We see the transformed dataframe without null values:

.. code:: python

        x1 x2  x3
    0  2.0  a   2
    2  1.0  b   4

By default, :class:`DropMissingData()` will find and store the columns that had
missing data during fit, that is, in the training set. They are stored here:

.. code:: python

    dmd.variables_

.. code:: python

    ['x1', 'x2']

That means that every time that we apply `transform()` to a new dataframe, the
transformer will remove rows with nan values only in those columns.

If we want to force :class:`DropMissingData()` to drop na across all columns, regardless
of whether they had nan values during fit, we need to set up the class like this:

.. code:: python

    dmd =  DropMissingData(missing_only=False)
    Xt = dmd.fit_transform(X)

Now, when we explore the paramter `variables_`, we see that all the variables in the
train set are stored, and hence, will be used to remove nan values:

.. code:: python

    dmd.variables_

.. code:: python

    ['x1', 'x2', 'x3']

Adjust target after dropna
^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`DropMissingData()` has the option to remove rows with nan from both training set
and target variable. Like this, we can obtain a target that is aligned with the
resulting dataframe after the transformation.

The method `transform_x_y` removes rows with null values from the train set, and then
realigns the target. Let's take a look:

.. code:: python

    Xt, yt = dmd.transform_x_y(X, y)
    Xt

Below we see the dataframe without nan:

.. code:: python

        x1 x2  x3
    0  2.0  a   2
    2  1.0  b   4

.. code:: python

    yt

And here we see the target with those rows corresponing to the remaining rows in the
transformed dataframe:

.. code:: python

    0    1
    2    3
    dtype: int64

Let's check that the shape of the transformed dataframe and target are the same:

.. code:: python

    Xt.shape, yt.shape

We see that the resulting training set and target have each 2 rows, instead of the 5
original rows.

.. code:: python

    ((2, 3), (2,))


Return the rows with nan
^^^^^^^^^^^^^^^^^^^^^^^^

When we have a model in production, it might be useful to know which rows are being
removed by the transformer. We can obtain that information as follows:

.. code:: python

    dmd.return_na_data(X)

The previous command returns the rows with nan. In other words, it does the opposite
of `transform()`, or pandas.dropna.

.. code:: python

        x1   x2  x3
    1  1.0  NaN   3
    3  0.0  NaN   5
    4  NaN    a   5


Dropna from subset of variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can choose to remove missing data only from a specific column or group of columns.
We just need to pass the column name or names to the `variables` parameter:

Here, we'll dropna from the variables "x1", "x3".

.. code:: python

    dmd = DropMissingData(variables=["x1", "x3"], missing_only=False)
    Xt = dmd.fit_transform(X)
    Xt.head()

Below, we see the transformed dataframe. It removed the rows with nan in "x1", and we
see that those rows with nan in "x2" are still in the dataframe:

.. code:: python

        x1   x2  x3
    0  2.0    a   2
    1  1.0  NaN   3
    2  1.0    b   4
    3  0.0  NaN   5

Only rows with nan in "x1" and "x3" are removed. We can corroborate that by examining
the `variables_` parameter:

.. code::python

    dmd.variables_

.. code::python

    ['x1', 'x3']

**Important**

When you indicate which variables should be examined to remove rows with nan, make sure
you set the parameter `missing_only` to the boolean `False`. Otherwise,
:class:`DropMissingData()` will select from your list only those variables that showed
nan values in the train set.

See for example what happens when we set up the class like this:

.. code:: python

    dmd = DropMissingData(variables=["x1", "x3"], missing_only=True)
    Xt = dmd.fit_transform(X)
    dmd.variables_

Note, that we indicated that we wanted to remove nan from "x1", "x3". Yet, only "x1"
has nan in X. So the transformer learns that nan should be only dropped from "x1":

.. code:: python

    ['x1']

:class:`DropMissingData()` took the 2 variables indicated in the list, and stored
only the one that showed nan in during fit. That means that when transforming future
dataframes, it will only remove rows with nan in "x1".

In other words, if you pass a list of variables to impute and set `missing_only=True`,
and some of the variables in your list do not have missing data in the train set,
missing data will not be removed during transform for those particular variables.

When `missing_only=True`, the transformer "double checks" that the entered
variables have missing data in the train set. If not, it ignores them during
`transform()`.

It is recommended to use `missing_only=True` when not passing a list of variables to
impute.

Dropna based on percentage of non-nan values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can set :class:`DropMissingData()` to require a percentage of non-NA values in a
row to keep it. We can control this behaviour through the `threshold` parameter, which
is equivalent to pandas.dropna's `thresh` parameter.

If `threshold=1`, all variables need to have data to keep a row. If `threshold=0.5`,
50% of the variables need to have data to keep a row. If `threshold=0.01`, 10% of the
variables need to have data to keep the row. If `threshold=None`, rows with NA in any
of the variables will be dropped.

Let's see this with an example. We create a new dataframe that has different proportion
of non-nan values in every row.

.. code:: python

    X = pd.DataFrame(
        dict(
            x1=[2, 1, 1, np.nan, np.nan],
            x2=["a", np.nan, "b", np.nan, np.nan],
            x3=[2, 3, 4, 5, np.nan],
        )
    )
    X

We see that the bottom row has nan in all columns, row 3 has nan in 2 of 3 columns,
and row 1 has nan in 1 variable:

.. code:: python

        x1   x2   x3
    0  2.0    a  2.0
    1  1.0  NaN  3.0
    2  1.0    b  4.0
    3  NaN  NaN  5.0
    4  NaN  NaN  NaN

Now, we can set :class:`DropMissingData()` to drop rows if >50% of its values are nan:

.. code:: python

    dmd = DropMissingData(threshold=.5)
    dmd.fit(X)
    dmd.transform(X)

We see that the last 2 rows are dropped, because they have more than 50% nan values.

.. code:: python

        x1   x2   x3
    0  2.0    a  2.0
    1  1.0  NaN  3.0
    2  1.0    b  4.0

Instead, we can set class:`DropMissingData()` to drop rows if >70% of its values are
nan as follows:

.. code:: python

    dmd = DropMissingData(threshold=.3)
    dmd.fit(X)
    dmd.transform(X)

Now we see that only the last row was removed.

.. code:: python

        x1   x2   x3
    0  2.0    a  2.0
    1  1.0  NaN  3.0
    2  1.0    b  4.0
    3  NaN  NaN  5.0

Scikit-learn compatible
^^^^^^^^^^^^^^^^^^^^^^^

:class:`DropMissingData()` is fully compatible with the Scikit-learn API, so you will
find common methods that you also find in Scikit-learn transformers, like, for example,
the `get_feature_names_out()` method to obtain the variable names in the transformed
dataframe.


Pipeline
^^^^^^^^

When we dropna from a dataframe, we then need to realign the target. We saw previously
that we can do that by using the method `transform_x_y`.

We can align the target with the resulting dataframe automatically from within a
pipeline as well, by utilizing Feature-engine's pipeline.

Let's start by importing the necessary libraries:

.. code:: python

    import numpy as np
    import pandas as pd

    from feature_engine.imputation import DropMissingData
    from feature_engine.encoding import OrdinalEncoder
    from feature_engine.pipeline import Pipeline

Let's create a new dataframe with nan values in some rows, two numerical and one
categorical variable, and its corresponding target variable:

.. code:: python

    X = pd.DataFrame(
        dict(
            x1=[2, 1, 1, 0, np.nan],
            x2=["a", np.nan, "b", np.nan, "a"],
            x3=[2, 3, 4, 5, 5],
        )
    )
    y = pd.Series([1, 2, 3, 4, 5])

    X.head()

Below, we see the resulting dataframe:

.. code:: python

        x1   x2  x3
    0  2.0    a   2
    1  1.0  NaN   3
    2  1.0    b   4
    3  0.0  NaN   5
    4  NaN    a   5

Let's now set up a pipeline to dropna first, and then encode the categorical variable
by using ordinal encoding:

.. code:: python

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", OrdinalEncoder(encoding_method="arbitrary")),
        ]
    )

    pipe.fit_transform(X, y)

When we apply `fit` and `transform` or `fit_transform`, we will obtain the transformed
training set only:

.. code:: python

        x1  x2  x3
    0  2.0   0   2
    2  1.0   1   4

To obtain the transform training set and target, we use `transform_x_y`:

.. code:: python

    pipe.fit(X,y)
    Xt, yt = pipe.transform_x_y(X, y)
    Xt

Here we see the transformed training set:

.. code:: python

        x1  x2  x3
    0  2.0   0   2
    2  1.0   1   4

.. code:: python

    yt

And here we see the re-aligned target variable:

.. code:: python

    0    1
    2    3

And to wrap up, let's add an estimator to the pipeline:

.. code:: python

    import numpy as np
    import pandas as pd

    from sklearn.linear_model import Lasso

    from feature_engine.imputation import DropMissingData
    from feature_engine.encoding import OrdinalEncoder
    from feature_engine.pipeline import Pipeline

    df = pd.DataFrame(
        dict(
            x1=[2, 1, 1, 0, np.nan],
            x2=["a", np.nan, "b", np.nan, "a"],
            x3=[2, 3, 4, 5, 5],
        )
    )
    y = pd.Series([1, 2, 3, 4, 5])

    pipe = Pipeline(
        [
            ("drop", DropMissingData()),
            ("enc", OrdinalEncoder(encoding_method="arbitrary")),
            ("lasso", Lasso(random_state=2))
        ]
    )

    pipe.fit(df, y)
    pipe.predict(df)

.. code:: python

    array([2., 2.])

Dropna or fillna?
^^^^^^^^^^^^^^^^^

:class:`DropMissingData()` has the same functionality than `pandas.series.dropna` or
`pandas.dataframe.dropna``. If you want functionality compatible with `pandas.fillna`
instead, check our other imputation transformers.


Drop columns with nan
^^^^^^^^^^^^^^^^^^^^^

At the moment, Feature-engine does not have transformers that will find columns with a
certain percentage of missing values and drop them. Instead, you can find those columns
manually, and then drop them with the help of `DropFeatures` from the selection module.

See also
^^^^^^^^

Check out our tutorials on `LagFeatures` and `WindowFeatures` to see how to combine
:class:`DropMissingData()` with lags or rolling windows, to create features for
forecasting.


Tutorials, books and courses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following Jupyter notebook, in our accompanying Github repository, you will find
more examples using :class:`DropMissingData()`.

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/imputation/DropMissingData.ipynb>`_

For tutorials about this and other feature engineering methods check out our online course:

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