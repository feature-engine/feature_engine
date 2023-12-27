.. _find_all_vars:

.. currentmodule:: feature_engine.variable_handling

find_all_variables
==================

With :class:`find_all_variables()` you can automatically capture in a list the names of
all the variables in the dataset.

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

We can now use :class:`find_all_variables()` to capture all the variable names in a list.
So let's do that and then display the items in the list:

.. code:: python

    from feature_engine.variable_handling import find_all_variables

    vars_all = find_all_variables(X)

    vars_all

We see the variable names in the list below:

.. code:: python

    ['num_var_1',
     'num_var_2',
     'num_var_3',
     'num_var_4',
     'cat_var1',
     'cat_var2',
     'date1',
     'date2',
     'date3']

We have the option to return the name of the variables of type categorical, object and
numerical only, or in other words, to exclude datetime variables. We can do so as
follows:

.. code:: python

    vars_all = find_all_variables(X, exclude_datetime=True)

    vars_all

In the list below, we can see that variables of type datetime were ignored:

.. code:: python

    ['num_var_1',
     'num_var_2',
     'num_var_3',
     'num_var_4',
     'cat_var1',
     'cat_var2',
     'date3']
