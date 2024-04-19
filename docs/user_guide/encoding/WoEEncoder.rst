.. _woe_encoder:

.. currentmodule:: feature_engine.encoding

WoEEncoder
==========

The :class:`WoEEncoder()` replaces categories by the weight of evidence
(WoE). The WoE was used primarily in the financial sector to create credit risk
scorecards.

The weight of evidence is given by:

.. math::

    log( p(X=xj|Y = 1) / p(X=xj|Y=0) )



**The WoE is determined as follows:**

We calculate the percentage positive cases in each category of the total of all
positive cases. For example 20 positive cases in category A out of 100 total
positive cases equals 20 %. Next, we calculate the percentage of negative cases in
each category respect to the total negative cases, for example 5 negative cases in
category A out of a total of 50 negative cases equals 10%. Then we calculate the
WoE by dividing the category percentages of positive cases by the category
percentage of negative cases, and take the logarithm, so for category A in our
example WoE = log(20/10).

**Note**

- If WoE values are negative, negative cases supersede the positive cases.
- If WoE values are positive, positive cases supersede the negative cases.
- And if WoE is 0, then there are equal number of positive and negative examples.

**Encoding into WoE**:

- Creates a monotonic relationship between the encoded variable and the target
- Returns variables in a similar scale

**Note**

This categorical encoding is exclusive for binary classification.

Let's look at an example using the Titanic Dataset.

First, let's load the data and separate it into train and test:


.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import WoEEncoder, RareLabelEncoder

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

Before we encode the variables, I would like to group infrequent categories into one
category, called 'Rare'. For this, I will use the :class:`RareLabelEncoder()` as follows:

.. code:: python

    # set up a rare label encoder
    rare_encoder = RareLabelEncoder(
        tol=0.1,
        n_categories=2,
        variables=['cabin', 'pclass', 'embarked'],
        ignore_format=True,
    )

    # fit and transform data
    train_t = rare_encoder.fit_transform(X_train)
    test_t = rare_encoder.transform(X_train)

Now, we set up the :class:`WoEEncoder()` to replace the categories by the weight of the
evidence, only in the 3 indicated variables:

.. code:: python

    # set up a weight of evidence encoder
    woe_encoder = WoEEncoder(
        variables=['cabin', 'pclass', 'embarked'],
        ignore_format=True,
    )

    # fit the encoder
    woe_encoder.fit(train_t, y_train)

With `fit()` the encoder learns the weight of the evidence for each category, which are stored in
its `encoder_dict_` parameter:

.. code:: python

	woe_encoder.encoder_dict_

In the `encoder_dict_` we find the WoE for each one of the categories of the
variables to encode. This way, we can map the original values to the new value.

.. code:: python

    {'cabin': {'M': -0.35752781962490193, 'Rare': 1.083797390800775},
     'pclass': {'1': 0.9453018143294478,
      '2': 0.21009172435857942,
      '3': -0.5841726684724614},
     'embarked': {'C': 0.679904786667102,
      'Rare': 0.012075414091446468,
      'S': -0.20113381737960143}}

Now, we can go ahead and encode the variables:

.. code:: python

    train_t = woe_encoder.transform(train_t)
    test_t = woe_encoder.transform(test_t)

    print(train_t.head())

Below we see the resulting dataset with the weight of the evidence:

.. code:: python

            pclass     sex        age  sibsp  parch     fare     cabin  embarked
    501   0.210092  female  13.000000      0      1  19.5000 -0.357528 -0.201134
    588   0.210092  female   4.000000      1      1  23.0000 -0.357528 -0.201134
    402   0.210092  female  30.000000      1      0  13.8583 -0.357528  0.679905
    1193 -0.584173    male  29.881135      0      0   7.7250 -0.357528  0.012075
    686  -0.584173  female  22.000000      0      0   7.7250 -0.357528  0.012075


**WoE for continuous variables**

In credit scoring, continuous variables are also transformed using the WoE. To do
this, first variables are sorted into a discrete number of bins, and then these
bins are encoded with the WoE as explained here for categorical variables. You can
do this by combining the use of the equal width, equal frequency or arbitrary
discretisers.

Additional resources
--------------------

In the following notebooks, you can find more details into the :class:`WoEEncoder()`
functionality and example plots with the encoded variables:

- `WoE in categorical variables <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/WoEEncoder.ipynb>`_
- `WoE in numerical variables <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/discretisation/EqualFrequencyDiscretiser_plus_WoEEncoder.ipynb>`_

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