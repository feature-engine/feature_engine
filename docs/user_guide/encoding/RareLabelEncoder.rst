.. _rarelabel_encoder:

.. currentmodule:: feature_engine.encoding

RareLabelEncoder
================

:class:`RareLabelEncoder()` groups infrequent categories into one new category
called 'Rare' or a different string indicated by the user.

This transformers requires 2 parameters:

- The minimum frequency a category should have not to be grouped.
- The minimum cardinality a variable should have to be processed.

Category frequency
~~~~~~~~~~~~~~~~~~

The parameter `tol` specifies the minimum proportion of observations a category must
have to remain ungrouped. In other words, categories with a frequency <= `tol` are
grouped into a single category.

Variable cardinality
~~~~~~~~~~~~~~~~~~~~

The parameter `n_categories` specifies the minimum cardinality a categorical variable must
have for infrequent categories to be grouped.

For example, if `n_categories = 5`, grouping is applied only to categorical variables
with more than five unique categories. Variables with five or fewer categories are left
unchanged.

.. tip::

    This parameter is useful for large datasets, where it may not be practical to examine all
    categorical variables individually. It ensures that variables with low cardinality are
    not reduced further.


Encoding popular categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter `max_n_categories` specifies the maximum number of unique categories
allowed in the encoded variable. If `max_n_categories = 5`, the five most frequent
categories are retained after encoding, and all others are grouped into a single category.

Python implementation
---------------------

Let's explore how to use :class:`RareLabelEncoder()` using the Titanic Dataset.

Let's load the data and separate it into train and test:

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

Let's explore the number of unique categories in the variable `"cabin"`.

.. code:: python

    X_train["cabin"].unique()

We see the number of unique categories in the output below:

.. code:: python

    array(['M', 'E', 'C', 'D', 'B', 'A', 'F', 'T', 'G'], dtype=object)

Now, we set up the :class:`RareLabelEncoder()` to group categories shown by less than 3%
of the observations into a new group called 'Rare'. We will group the
categories in the variables cabin, pclass and embarked, only if they have more than 2
unique categories each.

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
3% of the observations, that is, those that will not be grouped. These categories are stored
in the `encoder_dict_` attribute.

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

    import pandas as pd
    from feature_engine.encoding import RareLabelEncoder
    data = {'var_A': ['A'] * 10 + ['B'] * 10 + ['C'] * 2 + ['D'] * 1}
    data = pd.DataFrame(data)
    data['var_A'].value_counts()

In the following output, we see the number of observations per category:

.. code:: python

    A    10
    B    10
    C     2
    D     1
    Name: var_A, dtype: int64

Now, we group categories only for variables with more than 3 unique categories:

.. code:: python

    rare_encoder = RareLabelEncoder(tol=0.05, n_categories=3)
    rare_encoder.fit_transform(data)['var_A'].value_counts()

Note that the variable was left unchanged because it has exactly 3 unique categories:

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

In the following output we see that the 2 most infrequent categories have been grouped into
a new category called `Rare`:

.. code:: python

    A       10
    B       10
    Rare     3
    Name: var_A, dtype: int64

Considerations
--------------

:class:`RareLabelEncoder()` can be used to group infrequent categories and hence
control the expansion of the feature space if using one hot encoding.

Some categorical encodings will return NAN if a category is present in the test
set, but was not seen in the train set. This inconvenient can be mitigated if we
group rare labels before training the encoders.

Some categorical encoders will return NAN if there is not enough observations for
a certain category to calculate the mapping, for example :class:`WoEEncoder()`. These
type of errors can be prevented by grouping infrequent labels before the encoding with
:class:`RareLabelEncoder()`.


Additional resources
--------------------

For tutorials about this and other feature engineering methods check out these resources:

- `Feature Engineering for Machine Learning <https://www.trainindata.com/p/feature-engineering-for-machine-learning>`_, online course.
- `Feature Engineering for Time Series Forecasting <https://www.trainindata.com/p/feature-engineering-for-forecasting>`_, online course.
- `Python Feature Engineering Cookbook <https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587>`_, book.

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting `Sole <https://linkedin.com/in/soledad-galli>`_,
the main developer of feature-engine.