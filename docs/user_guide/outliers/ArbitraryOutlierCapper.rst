.. _arbitrary_capper:

.. currentmodule:: feature_engine.outliers

ArbitraryOutlierCapper
======================

The :class:`ArbitraryOutlierCapper()` caps the maximum or minimum values of a variable
at an arbitrary value indicated by the user. The maximum or minimum values should be
entered in a dictionary with the form {feature:capping value}.

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
variable 'age' at 50 and the variable 'Fare' at 200. We do not want to cap these variables
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

.. code:: python

    fare    200.0
    age      50.0
    dtype: float64


More details
^^^^^^^^^^^^

You can find more details about the :class:`ArbitraryOutlierCapper()` functionality in the following
notebook:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/outliers/ArbitraryOutlierCapper.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
