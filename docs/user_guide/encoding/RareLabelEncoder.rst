.. _rarelabel_encoder:

.. currentmodule:: feature_engine.encoding

RareLabelEncoder
================

The :class:`RareLabelEncoder()` groups infrequent categories into one new category
called 'Rare' or a different string indicated by the user. We need to specify the
minimum percentage of observations a category should have to be preserved and the
minimum number of unique categories a variable should have to be re-grouped.

**tol**

In the parameter `tol` we indicate the minimum proportion of observations a category
should have, not to be grouped. In other words, categories which frequency, or proportion
of observations is <= `tol` will be grouped into a unique term.

**n_categories**

In the parameter `n_categories` we indicate the minimum cardinality of the categorical
variable in order to group infrequent categories. For example, if `n_categories=5`,
categories will be grouped only in those categorical variables with more than 5 unique
categories. The rest of the variables will be ignored.

This parameter is useful when we have big datasets and do not have time to examine all
categorical variables individually. This way, we ensure that variables with low cardinality
are not reduced any further.

**max_n_categories**

In the parameter `max_n_categories` we indicate the maximum number of unique categories
that we want in the encoded variable. If `max_n_categories=5`, then the most popular 5
categories will remain in the variable after the encoding, all other will be grouped into
a single category.

This parameter is useful if we are going to perform one hot encoding at the back of it,
to control the expansion of the feature space.

**Example**

Let's look at an example using the Titanic Dataset.

First, let's load the data and separate it into train and test:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import RareLabelEncoder

    X, y = load_titanic(
        return_X_y_frame=True,
        handle_missing=True,
        predictors_only=True,
        cabin="letter_only",
    )
    X["pclass"] = X["pclass"].astype("O")

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

Let's explore the number of uniue categories in the variable `"cabin"`.

.. code:: python

    X_train["cabin"].unique()

We see the number of unique categories in the output below:

.. code:: python

    array(['M', 'E', 'C', 'D', 'B', 'A', 'F', 'T', 'G'], dtype=object)

Now, we set up the :class:`RareLabelEncoder()` to group categories shown by less than 3%
of the observations into a new group or category called 'Rare'. We will group the
categories in the indicated variables if they have more than 2 unique categories each.

.. code:: python

    encoder = RareLabelEncoder(
        tol=0.03,
        n_categories=2,
        variables=['cabin', 'pclass', 'embarked'],
        replace_with='Rare',
    )

    # fit the encoder
    encoder.fit(X_train)

With `fit()`, the :class:`RareLabelEncoder()` finds the categories present in more than
3% of the observations, that is, those that will not be grouped. These categories can
be found in the `encoder_dict_` attribute.

.. code:: python

    encoder.encoder_dict_

In the `encoder_dict_` we find the most frequent categories per variable to encode.
Any category that is not in this dictionary, will be grouped.

.. code:: python

    {'cabin': ['M', 'C', 'B', 'E', 'D'],
    'pclass': [3, 1, 2],
    'embarked': ['S', 'C', 'Q']}

Now we can go ahead and transform the variables:

.. code:: python

    # transform the data
    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

Let's now inspect the number of unique categories in the variable `"cabin"` after the
transformation:

.. code:: python

    train_t["cabin"].unique()

In the output below, we see that the infrequent categories have been replaced by
`"Rare"`.

.. code:: python

    array(['M', 'E', 'C', 'D', 'B', 'Rare'], dtype=object)

We can also specify the maximum number of categories that can be considered frequent
using the `max_n_categories` parameter.

Let's begin by creating a toy dataframe and count the values of observations per
category:

.. code:: python

    from feature_engine.encoding import RareLabelEncoder
    import pandas as pd
    data = {'var_A': ['A'] * 10 + ['B'] * 10 + ['C'] * 2 + ['D'] * 1}
    data = pd.DataFrame(data)
    data['var_A'].value_counts()

.. code:: python

    A    10
    B    10
    C     2
    D     1
    Name: var_A, dtype: int64

In this block of code, we group the categories only for variables with more than 3
unique categories and then we plot the result:

.. code:: python

    rare_encoder = RareLabelEncoder(tol=0.05, n_categories=3)
    rare_encoder.fit_transform(data)['var_A'].value_counts()

.. code:: python

    A       10
    B       10
    C        2
    Rare     1
    Name: var_A, dtype: int64

Now, we retain the 2 most frequent categories of the variable and group the rest into
the 'Rare' group:

.. code:: python

    rare_encoder = RareLabelEncoder(tol=0.05, n_categories=3, max_n_categories=2)
    Xt = rare_encoder.fit_transform(data)
    Xt['var_A'].value_counts()

.. code:: python

    A       10
    B       10
    Rare     3
    Name: var_A, dtype: int64

Tips
----

The :class:`RareLabelEncoder()` can be used to group infrequent categories and like this
control the expansion of the feature space if using one hot encoding.

Some categorical encodings will also return NAN if a category is present in the test
set, but was not seen in the train set. This inconvenient can usually be avoided if we
group rare labels before training the encoders.

Some categorical encoders will also return NAN if there is not enough observations for
a certain category. For example the :class:`WoEEncoder()` and the :class:`PRatioEncoder()`.
This behaviour can be also prevented by grouping infrequent labels before the encoding
with the :class:`RareLabelEncoder()`.


Additional resources
--------------------

In the following notebook, you can find more details into the :class:`RareLabelEncoder()`
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/RareLabelEncoder.ipynb>`_

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