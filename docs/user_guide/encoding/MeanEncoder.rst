.. _mean_encoder:

.. currentmodule:: feature_engine.encoding

MeanEncoder
===========

Mean encoding is the process of replacing the categories in categorical features by the
mean value of the target variable shown by each category. For example, if we are trying
to predict the default rate (that's the target variable), and our dataset has the categorical
variable **City**, with the categories of **London**, **Manchester**, and **Bristol**,
and the default rate per city is 0.1, 0.5, and 0.3, respectively, with mean encoding, we
would replace London by 0.1, Manchester by 0.5, and Bristol by 0.3.

Mean encoding, together with one hot encoding and ordinal encoding, belongs to the most
commonly used categorical encoding techniques in data science.

It is said that mean encoding can easily cause overfitting. That's because we are capturing
some information about the target into the predictive features during the encoding. More
importantly, the overfitting can be caused by encoding categories with low frequencies
with mean target values that are unreliable. In short, the mean target values seen for
those categories in the training set do not hold for test data or new observations.

Overfitting
-----------

When the categories in the categorical features have a good representation, or, in other
words, when there are enough observations in our dataset that show the categories that we
want to encode, then taking the simple average of the target variable per category is a
good approximation. We can trust that a new data point, say from the test data, that
shows that category will also have a target value that is similar to the target mean
value that we calculated for said category during training.

However, if there are only a few observations that show some of the categories, then the
mean target value for those categories will be unreliable. In other words, the certainty
that we have that a new observation that shows this category will have a mean target value
close to the one we estimated decreases.

To account for the uncertainty of the encoding values for rare categories, what we normally
do is **"blend"** the mean target variable per category with the general mean of the target,
calculated over the entire training dataset. And this blending is proportional to the
variability of the target within that category and the category frequency.

Smoothing
---------

To avoid overfitting, we can determine the mean target value estimates as a mixture of two
values: the mean target value per category (known as the posterior) and the mean target
value in the entire dataset (known as the prior).

The following formula shows the estimation of the mean target value with smoothing:

.. math::

    mapping = (w_i) posterior + (1-w_i) prior

The prior and posterior values are “blended” using a weighting factor (`wi`). This weighting
factor is a function of the category group size (`n_i`) and the variance of the target in
the data (`t`) and within the category (`s`):

.. math::

    w_i = n_i t / (s + n_i t)

When the category group is large, the weighing factor is close to 1, and therefore more
weight is given to the posterior (the mean of the target per category). When the category
group size is small, then the weight gets closer to 0, and more weight is given to the
prior (the mean of the target in the entire dataset).

In addition, if the variability of the target within that category is large, we also give
more weight to the prior, whereas if it is small, then we give more weight to the posterior.

In short, adding smoothing can help prevent overfitting in those cases where categorical
data have many infrequent categories or show high cardinality.

High cardinality
----------------

High cardinality refers to a high number of unique categories in the categorical features.
Mean encoding was specifically designed to tackle highly cardinal variables by taking
advantage of this smoothing function, which will essentially blend infrequent categories
together by replacing them with values very close to the overall target mean calculated
over the training data.

Another encoding method that tackles cardinality out of the box is count encoding. See for
example :class:`CountFrequencyEncoder`.

To account for highly cardinal variables in alternative encoding methods, you can group
rare categories together by using the :class:`RareLabelEncoder`.


Alternative Python implementations of mean encoding
---------------------------------------------------

In Feature-engine, we blend the probabilities considering the target variability and the
category frequency. In the original paper, there are alternative formulations to determine
the blending. If you want to check those out, use the transformers from the Python library
Category encoders:

- `M-estimate <https://contrib.scikit-learn.org/category_encoders/mestimate.html>`_
- `Target Encoder <https://contrib.scikit-learn.org/category_encoders/targetencoder.html>`_

Mean encoder
------------

Feature-engine's :class:`MeanEncoder()` replaces categories with the mean of the target per
category. By default, it does not implement smoothing. That means that it will replace
categories by the mean target value as determined during training over the training data
set (just the posterior).

To apply smoothing using the formulation that we described earlier, set the parameter
`smoothing` to `"auto"`. That would be our recommended solution. Alternatively, you can
set the parameter `smoothing` to any value that you want, in which case the weighting
factor `wi` will be calculated like this:

.. math::

    w_i = n_i / (s + n_i)

where s is the value your pass to `smoothing`.

Unseen categories
-----------------

Unseen categories are those labels that were not seen during training. Or in other words,
categories that were not present in the training data.

With the :class:`MeanEncoder()`, we can take care of unseen categories in 1 of 3 ways:

- We can set the mean encoder to ignore unseen categories, in which case those categories will be replaced by nan.
- We can set the mean encoder to raise an error when it encounters unseen categories. This is useful when we don't expect new categories for those categorical variables.
- We can instruct the mean encoder to replace unseen or new categories with the mean of the target shown in the training data, that is, the prior.

Mean encoding and machine learning
----------------------------------

Feature-engine's :class:`MeanEncoder()` can perform mean encoding for regression and binary
classification datasets. At the moment, we do not support multi-class targets.

Python examples
---------------

In the following sections, we'll show the functionality of :class:`MeanEncoder()` using the
Titanic Dataset.

First, let's load the libraries, functions and classes:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import MeanEncoder

To avoid data leakage, it is important to separate the data into training and test sets.
The mean target values, with or without smoothing, will be determined using the training
data only.

Let's load and split the data:

.. code:: python

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

We see the resulting dataframe containing 3 categorical columns: sex, cabin and embarked:

.. code:: python

          pclass     sex        age  sibsp  parch     fare cabin embarked
    501        2  female  13.000000      0      1  19.5000     M        S
    588        2  female   4.000000      1      1  23.0000     M        S
    402        2  female  30.000000      1      0  13.8583     M        C
    1193       3    male  29.881135      0      0   7.7250     M        Q
    686        3  female  22.000000      0      0   7.7250     M        Q


Simple mean encoding
--------------------

Let's set up the :class:`MeanEncoder()` to replace the categories in the categorical
features with the target mean, without smoothing:

.. code:: python

    encoder = MeanEncoder(
        variables=['cabin', 'sex', 'embarked'],
    )

    encoder.fit(X_train, y_train)

With `fit()` the encoder learns the target mean value for each category and stores those
values in the `encoder_dict_` attribute:

.. code:: python

   encoder.encoder_dict_

The `encoder_dict_` contains the mean value of the target per category, per variable.
We can use this dictionary to map the numbers in the encoded features to the original
categorical values.

.. code:: python

    {'cabin': {'A': 0.5294117647058824,
      'B': 0.7619047619047619,
      'C': 0.5633802816901409,
      'D': 0.71875,
      'E': 0.71875,
      'F': 0.6666666666666666,
      'G': 0.5,
      'M': 0.30484330484330485,
      'T': 0.0},
     'sex': {'female': 0.7283582089552239, 'male': 0.18760757314974183},
     'embarked': {'C': 0.553072625698324,
      'Missing': 1.0,
      'Q': 0.37349397590361444,
      'S': 0.3389570552147239}}

We can now go ahead and replace the categorical values with the numerical values:

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

Below we see the resulting dataframe, where the categorical values are now replaced
with the target mean values:

.. code:: python

          pclass       sex        age  sibsp  parch     fare     cabin  embarked
    501        2  0.728358  13.000000      0      1  19.5000  0.304843  0.338957
    588        2  0.728358   4.000000      1      1  23.0000  0.304843  0.338957
    402        2  0.728358  30.000000      1      0  13.8583  0.304843  0.553073
    1193       3  0.187608  29.881135      0      0   7.7250  0.304843  0.373494
    686        3  0.728358  22.000000      0      0   7.7250  0.304843  0.373494


Mean encoding with smoothing
----------------------------

By default, :class:`MeanEncoder()` determines the mean target values without blending.
If we want to apply smoothing to control the cardinality of the variable and avoid
overfitting, we set up the transformer as follows:

.. code:: python

    encoder = MeanEncoder(
        variables=None,
        smoothing="auto"
    )

    encoder.fit(X_train, y_train)

In this example, we did not indicate which variables to encode. :class:`MeanEncoder()` can
automatically find the categorical variables, which are stored in one of its attributes:

.. code:: python

    encoder.variables_

Below we see the categorical features found by :class:`MeanEncoder()`:

.. code:: python

    ['sex', 'cabin', 'embarked']

We can find the categorical mappings calculated by the mean encoder:

.. code:: python

   encoder.encoder_dict_

Note that these values are different to those determined without smoothing:

.. code:: python

    {'sex': {'female': 0.7275051072923914, 'male': 0.18782635616273297},
     'cabin': {'A': 0.5210189753697639,
      'B': 0.755161569137655,
      'C': 0.5608140829162441,
      'D': 0.7100896537503179,
      'E': 0.7100896537503179,
      'F': 0.6501082490288561,
      'G': 0.47606795923242295,
      'M': 0.3049458046855866,
      'T': 0.0},
     'embarked': {'C': 0.552100581239763,
      'Missing': 1.0,
      'Q': 0.3736336816011083,
      'S': 0.3390242994568531}}

We can now go ahead and replace the categorical values with the numerical values:

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

Below we see the resulting dataframe with the encoded features:

.. code:: python

          pclass       sex        age  sibsp  parch     fare     cabin  embarked
    501        2  0.727505  13.000000      0      1  19.5000  0.304946  0.339024
    588        2  0.727505   4.000000      1      1  23.0000  0.304946  0.339024
    402        2  0.727505  30.000000      1      0  13.8583  0.304946  0.552101
    1193       3  0.187826  29.881135      0      0   7.7250  0.304946  0.373634
    686        3  0.727505  22.000000      0      0   7.7250  0.304946  0.373634

We can now use this dataframes to train machine learning models for regression or
classification.

Mean encoding variables with numerical values
---------------------------------------------

:class:`MeanEncoder()`, and all Feature-engine encoders, have been designed to work with
variables of type object or categorical by default. If you want to encode variables that
are numeric, you need to instruct the transformer to ignore the data type:

.. code:: python

    encoder = MeanEncoder(
        variables=['cabin', 'pclass'],
        ignore_format=True,
    )

    t_train = encoder.fit_transform(X_train, y_train)
    t_test = encoder.transform(X_test)

After encoding the features we can use the data sets to train machine learning algorithms.

Last thing to note before closing in is that mean encoding does not increase the
dimensionality of the resulting dataframes: from 1 categorical feature, we obtain 1
encoded variable. Hence, this encoding method is suitable for predictive modeling that
uses models that are sensitive to the size of the feature space.

Additional resources
--------------------

In the following notebook, you can find more details into the :class:`MeanEncoder()`
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/MeanEncoder.ipynb>`_

For tutorials about this and other feature engineering methods check out these resources:


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

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.