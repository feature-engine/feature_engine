.. _check_all_vars:

.. currentmodule:: feature_engine.variable_handling

check_all_variables
===================

With :class:`check_all_variables()` we can check that the variables in a list are
present in the dataframe.

Let's create a toy dataset with numerical, categorical and datetime variables:

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform arrays into pandas df and series
    colnames = [f"num_var_{i+1}" for i in range(4)]
    X = pd.DataFrame(X, columns=colnames)

    X["cat_var1"] = ["Hello"] * 1000
    X["cat_var2"] = ["Bye"] * 1000

    X["date1"] = pd.date_range("2020-02-24", periods=1000, freq="T")
    X["date2"] = pd.date_range("2021-09-29", periods=1000, freq="H")
    X["date3"] = ["2020-02-24"] * 1000

    print(X.head())

We see the resulting dataframe below:

.. code:: python

       num_var_1  num_var_2  num_var_3  num_var_4 cat_var1 cat_var2  \
    0  -1.558594   1.634123   1.556932   2.869318    Hello      Bye
    1   1.499925   1.651008   1.159977   2.510196    Hello      Bye
    2   0.277127  -0.263527   0.532159   0.274491    Hello      Bye
    3  -1.139190  -1.131193   2.296540   1.189781    Hello      Bye
    4  -0.530061  -2.280109   2.469580   0.365617    Hello      Bye

                    date1               date2       date3
    0 2020-02-24 00:00:00 2021-09-29 00:00:00  2020-02-24
    1 2020-02-24 00:01:00 2021-09-29 01:00:00  2020-02-24
    2 2020-02-24 00:02:00 2021-09-29 02:00:00  2020-02-24
    3 2020-02-24 00:03:00 2021-09-29 03:00:00  2020-02-24
    4 2020-02-24 00:04:00 2021-09-29 04:00:00  2020-02-24


We can use :class:`check_all_variables()` with a list of variable names to verify that
the variables in the list are in the dataframe.

.. code:: python

    from feature_engine.variable_handling import check_all_variables

    checked_vars = check_all_variables(X, ["num_var_1", "cat_var1", "date1"])

    checked_vars

The output is the list of variable names passed to the function:

.. code:: python

    ['num_var_1', 'cat_var1', 'date1']

If we pass the name of a variable that is not in the dataframe, :class:`check_all_variables()`
will return an error:

.. code:: python

    check_all_variables(X, ["hola", "cat_var1", "date1"])

Below we see the error message:

.. code:: python

    KeyError: 'Some of the variables are not in the dataframe.'