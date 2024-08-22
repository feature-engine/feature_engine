.. _count_freq_encoder:

.. currentmodule:: feature_engine.encoding

CountFrequencyEncoder
=====================

Count encoding and frequency encoding are 2 categorical encoding techniques that were
commonly used during data preprocessing in Kaggle's data science competitions, even when
their predictive value is not immediately obvious.

Count encoding consists of replacing the categories of categorical features by their
counts, which are estimated from the training set. For example, in the variable color,
if 10 observations are blue and 5 observations are red, blue will be replaced by 10 and
red by 5.

Frequency encoding consists of replacing the labels of categorical data with their
frequency, which is also estimated from the training set. Then, in the variable City,
if London appears in 10% of the observations and Bristol in 1%, London will be replaced
by 0.1 and Bristol with 0.01.

Count and frequency encoding in machine learning
------------------------------------------------

We'd use count encoding or frequency encoding when we think that the representation of
the categories in the dataset has some sort of predictive value. To be honest, the only
example that I can think of where count encoding could be useful is in sales forecasting
or sales data analysis scenarios, where the count of a product or an item represents its
popularity. In other words, we may be more likely to sell a product with a high count.

Count encoding and frequency encoding can be suitable for categorical variables with high
cardinality because these types of categorical encoding will cause what is called
collisions: categories that are present in a similar number of observations will be
replaced with similar, if not the same values, which reduces the variability.

This, of course, can result in the loss of information by placing two categories that
are otherwise different in the same pot. But on the other hand, if we are using count
encoding or frequency encoding, we have reasons to believe that the count or the frequency
are a good indicator of predictive performance or somehow capture data insight, so that
categories with similar counts would show similar patterns or behaviors.

Count and Frequency encoding with Feature-engine
------------------------------------------------

The :class:`CountFrequencyEncoder()` replaces categories of categorical features by
either the count or the percentage of observations each category shows in the training set.

With :class:`CountFrequencyEncoder()` we can automatically encode all categorical
features in the dataset, or only a subset of them, by passing the variable names in a
list to the `variables` parameter when we set up the encoder.

By default, :class:`CountFrequencyEncoder()` will encode only categorical data. If we
want to encode numerical values, we need to explicitly say so by setting the parameter
`ignore_format` to True.

Count and frequency encoding with unseen categories
---------------------------------------------------

When we learn mappings from strings to numbers, either with count encoding or other
encoding techniques like ordinal encoding or target encoding, we do so by observing the
categories in the training set. Hence, we won't have mappings for categories that appear
only in the test set. These are the so-called "unseen categories."

When encountering unseen categories, :class:`CountFrequencyEncoder()` will ignore them
by default, which means that unseen categories will be replaced with missing values.
We can instruct the encoder to raise an error when a new category is encountered, or
alternatively, to encode unseen categories with zero.

Count encoding vs other encoding methods
----------------------------------------

Count and frequency encoding, similar to ordinal encoding and contrarily to one-hot
encoding, feature hashing, or binary encoding, does not increase the dataset dimensionality.
From one categorical variable, we obtain one numerical feature.

Python example
--------------

Let's examine the functionality of :class:`CountFrequencyEncoder()` by using the Titanic
dataset. We'll start by loading the libraries and functions, loading the dataset, and then
splitting it into a training and a testing set.

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import CountFrequencyEncoder

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

We see the resulting dataframe with the predictor variables below:

.. code:: python

          pclass     sex        age  sibsp  parch     fare cabin embarked
    501        2  female  13.000000      0      1  19.5000     M        S
    588        2  female   4.000000      1      1  23.0000     M        S
    402        2  female  30.000000      1      0  13.8583     M        C
    1193       3    male  29.881135      0      0   7.7250     M        Q
    686        3  female  22.000000      0      0   7.7250     M        Q

This dataset has three obvious categorical features: cabin, embarked, and sex, and in
addition, pclass could also be handled as a categorical.

Count encoding
~~~~~~~~~~~~~~

We'll start by encoding the three categorical variables using their counts, that is,
replacing the strings with the number of times each category is present in the training
dataset.

.. code:: python

    encoder = CountFrequencyEncoder(
    encoding_method='count',
    variables=['cabin', 'sex', 'embarked'],
    )

    encoder.fit(X_train)

With `fit()`, the count encoder learns the counts of each category. We can inspect the
counts as follows:

.. code:: python

    encoder.encoder_dict_

We see the counts of each category for each of the three variables in the following output:

.. code:: python


    {'cabin': {'M': 702,
      'C': 71,
      'B': 42,
      'E': 32,
      'D': 32,
      'A': 17,
      'F': 15,
      'G': 4,
      'T': 1},
     'sex': {'male': 581, 'female': 335},
     'embarked': {'S': 652, 'C': 179, 'Q': 83, 'Missing': 2}}

Now, we can go ahead and encode the variables:

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

We see the resulting dataframe where the categorical features are now replaced with
integer values corresponding to the category counts:

.. code:: python

          pclass  sex        age  sibsp  parch     fare  cabin  embarked
    501        2  335  13.000000      0      1  19.5000    702       652
    588        2  335   4.000000      1      1  23.0000    702       652
    402        2  335  30.000000      1      0  13.8583    702       179
    1193       3  581  29.881135      0      0   7.7250    702        83
    686        3  335  22.000000      0      0   7.7250    702        83

We can now use the encoded dataframes to train machine learning models.

Frequency encoding
~~~~~~~~~~~~~~~~~~

Let's now perform frequency encoding. We'll encode 2 categorical and 1 numerical variable,
hence, we need to set the encoder to ignore the variable's type:

.. code:: python

    encoder = CountFrequencyEncoder(
    encoding_method='frequency',
    variables=['cabin', 'pclass', 'embarked'],
    ignore_format=True,
    )


Now, we fit the frequency encoder to the train set and transform it straightaway, and
then we transform the test set:

.. code:: python

    t_train = encoder.fit_transform(X_train)
    t_test = encoder.transform(X_test)

    test.head()

In the following output we see the transformed dataframe, where the categorical features
are now encoded into their frequencies:

.. code:: python

            pclass     sex        age  sibsp  parch     fare     cabin  embarked
    1139  0.543668    male  38.000000      0      0   7.8958  0.766376   0.71179
    533   0.205240  female  21.000000      0      1  21.0000  0.766376   0.71179
    459   0.205240    male  42.000000      1      0  27.0000  0.766376   0.71179
    1150  0.543668    male  29.881135      0      0  14.5000  0.766376   0.71179
    393   0.205240    male  25.000000      0      0  31.5000  0.766376   0.71179

With `fit()` the encoder learns the frequencies of each category, which are stored in
its `encoder_dict_` parameter. We can inspect them like this:

.. code:: python

   encoder.encoder_dict_

In the `encoder_dict_` we find the frequencies for each one of the unique categories of
each variable to encode. This way, we can map the original value to the new value.

.. code:: python

   {'cabin': {'M': 0.7663755458515283,
      'C': 0.07751091703056769,
      'B': 0.04585152838427948,
      'E': 0.034934497816593885,
      'D': 0.034934497816593885,
      'A': 0.018558951965065504,
      'F': 0.016375545851528384,
      'G': 0.004366812227074236,
      'T': 0.001091703056768559},
   'pclass': {3: 0.5436681222707423,
      1: 0.25109170305676853,
      2: 0.2052401746724891},
   'embarked': {'S': 0.7117903930131004,
      'C': 0.19541484716157206,
      'Q': 0.0906113537117904,
      'Missing': 0.002183406113537118}}

We can now use these dataframes to train machine learning algorithms.

With the method `inverse_transform`, we can transform the encoded dataframes back to their
original representation, that is, we can replace the encoding with the original categorical
values.

Additional resources
--------------------

In the following notebook, you can find more details into the :class:`CountFrequencyEncoder()`
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/CountFrequencyEncoder.ipynb>`_

For more details about this and other feature engineering methods check out these resources:


.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

.. figure::  ../../images/fetsf.png
   :width: 300
   :figclass: align-center
   :align: right
   :target: https://www.trainindata.com/p/feature-engineering-for-forecasting

   Feature Engineering for Time Series Forecasting


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

Our book:

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

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.