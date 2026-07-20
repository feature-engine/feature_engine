.. _arbitrary_capper:

.. currentmodule:: feature_engine.outliers

ArbitraryOutlierCapper
======================

:class:`ArbitraryOutlierCapper()` caps the maximum or minimum values of a variable
at an arbitrary value indicated by the user. The maximum or minimum values should be
entered in a dictionary with the form {feature:capping value}.

Python implementation
---------------------

Let's look at this in an example. First we load the Titanic dataset, and separate it
into a train and a test set:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.outliers import ArbitraryOutlierCapper

    X, y = load_titanic(
        return_X_y_frame=True,
        predictors_only=True,
        handle_missing=True,
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )

    print(X_train.head())

We see the resulting data below:

.. code:: python

          pclass     sex        age  sibsp  parch     fare    cabin embarked
    501        2  female  13.000000      0      1  19.5000  Missing        S
    588        2  female   4.000000      1      1  23.0000  Missing        S
    402        2  female  30.000000      1      0  13.8583  Missing        C
    1193       3    male  29.881135      0      0   7.7250  Missing        Q
    686        3  female  22.000000      0      0   7.7250  Missing        Q

Now, we set up the :class:`ArbitraryOutlierCapper()` indicating that we want to cap the
variable 'age' at 50 and the variable 'fare' at 200. We do not want to cap these variables
on the left side of their distribution.

.. code:: python

    capper = ArbitraryOutlierCapper(
        max_capping_dict={'age': 50, 'fare': 200},
        min_capping_dict=None,
    )

    capper.fit(X_train)

With `fit()` the transformer does not learn any parameter. It just reassigns the entered
dictionary to the attribute that will be used in the transformation:

.. code:: python

    capper.right_tail_caps_

In the following output, we see that the dictionary we entered when setting up the transformer
was assigned to a different attribute after fitting:

.. code:: python

    {'age': 50, 'fare': 200}

Now, we can go ahead and cap the variables:

.. code:: python

    train_t = capper.transform(X_train)
    test_t = capper.transform(X_test)

If we now check the maximum values in the transformed data, they should be those entered
in the dictionary:

.. code:: python

    train_t[['fare', 'age']].max()

In the following output, we see that the variables were capped at the requested maximum
values:

.. code:: python

    fare    200.0
    age      50.0
    dtype: float64


Additional resources
--------------------

For more details about this and other feature engineering methods check out these resources:

- `Feature Engineering for Machine Learning <https://www.trainindata.com/p/feature-engineering-for-machine-learning>`_, online course.
- `Feature Engineering for Time Series Forecasting <https://www.trainindata.com/p/feature-engineering-for-forecasting>`_, online course.
- `Python Feature Engineering Cookbook <https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587>`_, book.

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting `Sole <https://linkedin.com/in/soledad-galli>`_,
the main developer of feature-engine.