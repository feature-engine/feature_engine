.. _winsorizer:

.. currentmodule:: feature_engine.outliers

Winsorizer
==========

The :class:`Winsorizer()` caps maximum and/or minimum values of a variable at automatically
determined values. The minimum and maximum values can be calculated in 1 of 3 different ways:

Gaussian limits:

- right tail: mean + 3* std
- left tail: mean - 3* std

IQR limits:

- right tail: 75th quantile + 1.5* IQR
- left tail:  25th quantile - 1.5* IQR

where IQR is the inter-quartile range: 75th quantile - 25th quantile.

MAD limits:

    - right tail: median + 3.29* MAD
    - left tail:  median - 3.29* MAD

where MAD is the median absolute deviation from the median.

percentiles or quantiles:

- right tail: 95th percentile
- left tail:  5th percentile

**Example**

Let's cap some outliers in the Titanic Dataset. First, let's load the data and separate
it into train and test:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.outliers import Winsorizer

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

Now, we will set the :class:`Winsorizer()` to cap outliers at the right side of the
distribution only (param `tail`). We want the maximum values to be determined using the
mean value of the variable (param `capping_method`) plus 3 times the standard deviation
(param `fold`). And we only want to cap outliers in 2 variables, which we indicate in a
list.

.. code:: python

    capper = Winsorizer(capping_method='gaussian',
                        tail='right', 
                        fold=3, 
                        variables=['age', 'fare'])

    capper.fit(X_train)

With `fit()`, the :class:`Winsorizer()` finds the values at which it should cap the variables.
These values are stored in its attribute:

.. code:: python

    capper.right_tail_caps_

.. code:: python

	{'age': 67.73951212364803, 'fare': 174.70395336846678}

We can now go ahead and censor the outliers:

.. code:: python

    # transform the data
    train_t = capper.transform(X_train)
    test_t = capper.transform(X_test)
    
If we evaluate now the maximum of the variables in the transformed datasets, they should
coincide with the values observed in the attribute `right_tail_caps_`:

.. code:: python

    train_t[['fare', 'age']].max()

.. code:: python

    fare    174.703953
    age      67.739512
    dtype: float64

Setting up the stringency (param `fold`)
----------------------------------------

By default, :class:`Winsorizer()` automatically determines the parameter `fold` based
on the chosen `capping_method`. This parameter determines the multiplier for standard
deviation, interquartile range (IQR), or Median Absolute Deviation (MAD), or
sets the percentile at which to cap the variables.

The default values for fold are as follows:

- 'gaussian': `fold` is set to 3.0;
- 'iqr': `fold` is set to 1.5;
- 'mad': `fold` is set to 3.29;
- 'percentiles': `fold` is set to 0.05.

You can manually adjust the `fold` value to make the outlier detection process more or less
conservative, thus customizing the extent of outlier capping.

Additional resources
--------------------

You can find more details about the :class:`Winsorizer()` functionality in the following
notebook:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/outliers/Winsorizer.ipynb>`_

For more details about this and other feature engineering methods check out these resources:


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