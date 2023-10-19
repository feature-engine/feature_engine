.. _onehot_encoder:

.. currentmodule:: feature_engine.encoding


OneHotEncoder
=============

The :class:`OneHotEncoder()` performs one hot encoding. One hot encoding consists in
replacing the categorical variable by a group of binary variables which take value 0 or
1, to indicate if a certain category is present in an observation. The binary variables
are also known as dummy variables.

For example, from the categorical variable "Gender" with categories "female" and
"male", we can generate the boolean variable "female", which takes 1 if the
observation is female or 0 otherwise. We can also generate the variable "male",
which takes 1 if the observation is "male" and 0 otherwise. By default, the
:class:`OneHotEncoder()` will return both binary variables from "Gender": "female" and
"male".

**Binary variables**

When a categorical variable has only 2 categories, like "Gender" in our previous example, then
the second dummy variable created by one hot encoding can be completely redundant. We
can drop automatically the last dummy variable for those variables that contain only 2
categories by setting the parameter `drop_last_binary=True`. This will ensure that for
every binary variable in the dataset, only 1 dummy is created. This is recommended,
unless we suspect that the variable could, in principle take more than 2 values.

**k vs k-1 dummies**

From a categorical variable with k unique categories, the :class:`OneHotEncoder()` can
create k binary variables, or alternatively k-1 to avoid redundant information. This
behaviour can be specified using the parameter `drop_last`. Only k-1 binary variables
are necessary to encode all of the information in the original variable. However, there
are situations in which we may choose to encode the data into k dummies.

Encode into k-1 if training linear models: Linear models evaluate all features during
fit, thus, with k-1 they have all information about the original categorical variable.

Encode into k if training decision trees or performing feature selection: tree based
models and many feature selection algorithms evaluate variables or groups of variables
separately. Thus, if encoding into k-1, the last category will not be examined. That is,
we lose the information contained in that category.

**Encoding only popular categories**

The encoder can also create binary variables for the n most popular categories, n being
determined by the user. For example, if we encode only the 6 more popular categories, by
setting the parameter `top_categories=6`, the transformer will add binary variables only
for the 6 most frequent categories. The most frequent categories are those with the biggest
number of observations. The remaining categories will not be encoded into dummies. Thus,
if an observation presents a category other than the most frequent ones, it will have a
0 value in each one of the derived dummies. This behaviour is useful when the categorical
variables are highly cardinal, to control the expansion of the feature space.

**Note**

Only when creating binary variables for all categories of the variable (instead of the
most popular ones), we can specify if we want to encode into k or k-1 binary variables,
where k is the number if unique categories. If we encode only the top n most popular
categories, the encoder will create only n binary variables per categorical variable.
Observations that do not show any of these popular categories, will have 0 in all
the binary variables.

Let's look at an example using the Titanic Dataset. First we load the data and divide it
into a train and a test set:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import OneHotEncoder

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

Now, we set up the encoder to encode only the 2 most frequent categories of 3
categorical variables:

.. code:: python

    encoder = OneHotEncoder(
        top_categories=2,
        variables=['pclass', 'cabin', 'embarked'],
        ignore_format=True,
        )

    # fit the encoder
    encoder.fit(X_train)

With `fit()` the encoder will learn the most popular categories of the variables, which
are stored in the attribute `encoder_dict_`.

.. code:: python

	encoder.encoder_dict_

.. code:: python

	{'pclass': [3, 1], 'cabin': ['M', 'C'], 'embarked': ['S', 'C']}

The `encoder_dict_` contains the categories that will derive dummy variables for each
categorical variable.

With transform, we go ahead and encode the variables. Note that by default, the
:class:`OneHotEncoder()` will drop the original variables.

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

Below we see the one hot dummy variables added to the dataset, and the original
variables were removed:

.. code:: python

             sex        age  sibsp  parch     fare  pclass_3  pclass_1  cabin_M  \
    501   female  13.000000      0      1  19.5000         0         0        1
    588   female   4.000000      1      1  23.0000         0         0        1
    402   female  30.000000      1      0  13.8583         0         0        1
    1193    male  29.881135      0      0   7.7250         1         0        1
    686   female  22.000000      0      0   7.7250         1         0        1

          cabin_C  embarked_S  embarked_C
    501         0           1           0
    588         0           1           0
    402         0           0           1
    1193        0           0           0
    686         0           0           0

If you do not want to drop the original variables, consider using the OneHotEncoder
from Scikit-learn.

**Feature space and duplication**

If the categorical variables are highly cardinal, we may end up with very big datasets
after one hot encoding. In addition, if some of these variables are fairly constant or
fairly similar, we may end up with one hot encoded features that are highly correlated
if not identical.

Consider checking this up and dropping redundant features with the transformers from the
:ref:`selection module <selection_user_guide>`.

More details
^^^^^^^^^^^^

For more details into :class:`OneHotEncoder()`'s functionality visit:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/OneHotEncoder.ipynb>`_

For more details about this and other feature engineering methods check out these resources:

- `Feature engineering for machine learning <https://www.trainindata.com/p/feature-engineering-for-machine-learning>`_, online course.
- `Python Feature Engineering Cookbook <https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming-dp-1804611301/dp/1804611301>`_, book.
