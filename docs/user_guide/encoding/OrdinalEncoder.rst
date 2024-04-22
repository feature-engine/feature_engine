.. _ordinal_encoder:

.. currentmodule:: feature_engine.encoding

OrdinalEncoder
==============


The :class:`OrdinalEncoder()` replaces the categories by digits, starting from 0 to k-1,
where k is the number of different categories. If you select **"arbitrary"** in the
`encoding_method`, then the encoder will assign numbers as the labels appear in the
variable (first come first served). If you select **"ordered"**, the encoder will assign
numbers following the mean of the target value for that label. So labels for which the
mean of the target is higher will get the number 0, and those where the mean of the
target is smallest will get the number k-1. This way, we create a monotonic relationship
between the encoded variable and the target.

**Arbitrary vs ordered encoding**

**Ordered ordinal encoding**: for the variable colour, if the mean of the target
for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 1,
red by 2 and grey by 0.

The motivation is to try and create a monotonic relationship between the target and the
encoded categories. This tends to help improve performance of linear models.

**Arbitrary ordinal encoding**: the numbers will be assigned arbitrarily to the
categories, on a first seen first served basis.

Let's look at an example using the Titanic Dataset.

First, let's load the data and separate it into train and test:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import OrdinalEncoder

    X, y = load_titanic(
        return_X_y_frame=True,
        handle_missing=True,
        predictors_only=True,
        cabin="letter_only",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )

    print(X_train.head())

We see the resulting data below:

.. code:: python

          pclass     sex        age  sibsp  parch     fare cabin embarked
    501        2  female  13.000000      0      1  19.5000     M        S
    588        2  female   4.000000      1      1  23.0000     M        S
    402        2  female  30.000000      1      0  13.8583     M        C
    1193       3    male  29.881135      0      0   7.7250     M        Q
    686        3  female  22.000000      0      0   7.7250     M        Q


Now, we set up the :class:`OrdinalEncoder()` to replace the categories by strings based
on the target mean value and only in the 3 indicated variables:

.. code:: python

    encoder = OrdinalEncoder(
        encoding_method='ordered',
        variables=['pclass', 'cabin', 'embarked'],
        ignore_format=True)

    encoder.fit(X_train, y_train)

With `fit()` the encoder learns the mappings for each category, which are stored in
its `encoder_dict_` parameter:

.. code:: python

	encoder.encoder_dict_

In the `encoder_dict_` we find the integers that will replace each one of the categories
of each variable that we want to encode. This way, we can map the original value of the
variable to the new value.

.. code:: python

	{'pclass': {3: 0, 2: 1, 1: 2},
	'cabin': {'T': 0,
		'M': 1,
		'G': 2,
		'A': 3,
		'C': 4,
		'F': 5,
		'D': 6,
		'E': 7,
		'B': 8},
	'embarked': {'S': 0, 'Q': 1, 'C': 2, 'Missing': 3}}

We can now go ahead and replace the original strings with the numbers:

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

Below we see the resulting dataframe, where the original variable values are now replaced
with integers:

.. code:: python

          pclass     sex        age  sibsp  parch     fare  cabin  embarked
    501        1  female  13.000000      0      1  19.5000      1         0
    588        1  female   4.000000      1      1  23.0000      1         0
    402        1  female  30.000000      1      0  13.8583      1         2
    1193       0    male  29.881135      0      0   7.7250      1         1
    686        0  female  22.000000      0      0   7.7250      1         1


Additional resources
--------------------

In the following notebook, you can find more details into the :class:`OrdinalEncoder()`'s
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/OrdinalEncoder.ipynb>`_

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
   :target: https://packt.link/0ewSo

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