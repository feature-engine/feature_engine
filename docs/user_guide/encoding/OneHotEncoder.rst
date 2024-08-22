.. _onehot_encoder:

.. currentmodule:: feature_engine.encoding


OneHotEncoder
=============

One-hot encoding is a method used to represent categorical data, where each category
is represented by a binary variable. The binary variable takes the value 1 if the
category is present and 0 otherwise. The binary variables are also known as dummy
variables.

To represent the categorical feature "is-smoker" with categories "Smoker" and
"Non-smoker", we can generate the dummy variable "Smoker", which takes 1 if the
person smokes and 0 otherwise. We can also generate the variable "Non-smoker", which
takes 1 if the person does not smoke and 0 otherwise.

The following table shows a possible one hot encoded representation of the variable
"is smoker":

============= ========== =============
  is smoker     smoker     non-smoker
============= ========== =============
smoker            1           0
non-smoker  	  0           1
non-smoker        0           1
smoker            1	          0
non-smoker	      0           1
============= ========== =============

For the categorical variable **Country** with values **England**, **Argentina**, and
**Germany**, we can create three variables called `England`, `Argentina`, and `Germany`.
These variables will take the value of 1 if the observation is England, Argentina, or
Germany, respectively, and 0 otherwise.

Encoding into k vs k-1 variables
--------------------------------

A categorical feature with k unique categories can be encoded using k-1 binary variables.
For `Smoker`, k is 2 as it contains two labels (Smoker and Non-Smoker), so we only
need one binary variable (k - 1 = 1) to capture all of the information.

In the following table we see that the dummy variable `Smoker` fully represents the
original categorical values:


============= ==========
  is smoker     smoker
============= ==========
smoker            1
non-smoker  	  0
non-smoker        0
smoker            1
non-smoker	      0
============= ==========

For the **Country** variable, which has three categories (k=3; England, Argentina, and
Germany), we need two (k - 1 = 2) binary variables to capture all the information. The
variable will be fully represented like this:


============= ========== =============
  Country      England     Argentina
============= ========== =============
England            1           0
Argentina  	       0           1
Germany            0           0
============= ========== =============

As we see in the previous table, if the observation is England, it will show the value 1 in
the `England` variable; if the observation is Argentina, it will show the value 1 in
the `Argentina` variable; and if the observation is Germany, it will show zeroes in
both dummy variables.

Like these, by looking at the values of the k-1 dummies, we can infer the original
categorical value of each observation.

Encoding into k-1 binary variables is well-suited for linear regression models. Linear
models evaluate all features during fit, thus, with k-1 they have all the information
about the original categorical variable.

There are a few occasions in which we may prefer to encode the categorical variables
with k binary variables.

Encode into k dummy variables if training decision trees based models or performing
feature selection. Decision tree based models and many feature selection algorithms
evaluate variables or groups of variables separately. Thus, if encoding into k-1, the
last category will not be examined. In other words, we lose the information contained
in that category.

Binary variables
----------------

When a categorical variable has only 2 categories, like "Smoker" in our previous example,
then encoding into k-1 suits all purposes, because the second dummy variable created
by one hot encoding is completely redundant.


Encoding popular categories
---------------------------

One hot encoding can increase the feature space dramatically, particularly if we have
many categorical features, or the features have high cardinality. To control the feature
space, it is common practice to encode only the most frequent categories in each
categorical variable.

When we encode the most frequent categories, we will create binary variables for each
of these frequent categories, and when the observation has a different, less popular
category, it will have a 0 in all binary variables. See the following example:

============== ========== =============
  var           popular1     popular2
============== ========== =============
popular1            1           0
popular2            0           1
popular1            1           0
non-popular         0           0
popular2	        0           1
less popular        0           0
unpopular           0           0
lonely              0           0
============== ========== =============

As we see in the previous table, less popular categories are represented as a group by
showing zeroes in all binary variables.

OneHotEncoder
-------------

Feature-engine's :class:`OneHotEncoder()` encodes categorical data as a one-hot numeric
dataframe.

:class:`OneHotEncoder()` can encode into k or k-1 dummy variables. The behaviour is
specified through the `drop_last` parameter, which can be set to `False` for k, or to
`True` for k-1 dummy variables.

:class:`OneHotEncoder()` can specifically encode binary variables into k-1 variables
(that is, 1 dummy) while encoding categorical features of higher cardinality into k
dummies. This behaviour is specified by setting the parameter `drop_last_binary=True`.
This will ensure that for every binary variable in the dataset, that is, for every
categorical variable with ONLY 2 categories, only 1 dummy is created. This is recommended,
unless you suspect that the variable could, in principle, take more than 2 values.

:class:`OneHotEncoder()` can also create binary variables for the **n** most popular
categories, n being determined by the user. For example, if we encode only the 6 more
popular categories, by setting the parameter `top_categories=6`, the transformer will
add binary variables only for the 6 most frequent categories. The most frequent categories
are those with the greatest number of observations. The remaining categories will show
zeroes in each one of the derived dummies. This behaviour is useful when the categorical
variables are highly cardinal to control the expansion of the feature space.

**Note**

The parameter `drop_last` is ignored when encoding the most popular categories.


Python implementation
---------------------

Let's look at an example of one hot encoding, using Feature-engine's  :class:`OneHotEncoder()`
utilizing the Titanic Dataset.

We'll start by importing the libraries, functions and classes, and loading the data into
a pandas dataframe and dividing it into a training and a testing set:

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

We see the first 5 rows of the training data below:

.. code:: python

          pclass     sex        age  sibsp  parch     fare cabin embarked
    501        2  female  13.000000      0      1  19.5000     M        S
    588        2  female   4.000000      1      1  23.0000     M        S
    402        2  female  30.000000      1      0  13.8583     M        C
    1193       3    male  29.881135      0      0   7.7250     M        Q
    686        3  female  22.000000      0      0   7.7250     M        Q

Let's explore the cardinality of 4 of the categorical features:

.. code:: python

    X_train[['sex', 'pclass', 'cabin', 'embarked']].nunique()

.. code:: python

    sex         2
    pclass      3
    cabin       9
    embarked    4
    dtype: int64

We see that the variable sex has 2 categories, pclass has 3 categories, the variable
cabin has 9 categories, and the variable embarked has 4 categories.

Let's now set up the OneHotEncoder to encode 2 of the categorical variables into k-1 dummy
variables:

.. code:: python

    encoder = OneHotEncoder(
        variables=['cabin', 'embarked'],
        drop_last=True,
        )

    encoder.fit(X_train)

With `fit()` the encoder learns the categories of the variables, which are stored in the
attribute `encoder_dict_`.

.. code:: python

   encoder.encoder_dict_

.. code:: python

    {'cabin': ['M', 'E', 'C', 'D', 'B', 'A', 'F', 'T'],
     'embarked': ['S', 'C', 'Q']}

The `encoder_dict_` contains the categories that will be represented by dummy variables
for each categorical variable.

With transform, we go ahead and encode the variables. Note that by default, the
:class:`OneHotEncoder()` drops the original categorical variables, which are now
represented by the one-hot array.

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

Below we see the one hot dummy variables added to the dataset and the original variables
are no longer in the dataframe:

.. code:: python

          pclass     sex        age  sibsp  parch     fare  cabin_M  cabin_E  \
    501        2  female  13.000000      0      1  19.5000        1        0
    588        2  female   4.000000      1      1  23.0000        1        0
    402        2  female  30.000000      1      0  13.8583        1        0
    1193       3    male  29.881135      0      0   7.7250        1        0
    686        3  female  22.000000      0      0   7.7250        1        0

          cabin_C  cabin_D  cabin_B  cabin_A  cabin_F  cabin_T  embarked_S  \
    501         0        0        0        0        0        0           1
    588         0        0        0        0        0        0           1
    402         0        0        0        0        0        0           0
    1193        0        0        0        0        0        0           0
    686         0        0        0        0        0        0           0

          embarked_C  embarked_Q
    501            0           0
    588            0           0
    402            1           0
    1193           0           1
    686            0           1


Finding categorical variables automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature-engine's :class:`OneHotEncoder()` can automatically find and encode all
categorical features in the pandas dataframe. Let's show that with an example.

Let's set up the OneHotEncoder to find and encode all categorical features:

.. code:: python

    encoder = OneHotEncoder(
        variables=None,
        drop_last=True,
        )

    encoder.fit(X_train)

With fit, the encoder finds the categorical features and identifies it's unique
categories. We can find the categorical variables like this:


.. code:: python

    encoder.variables_

.. code:: python

    ['sex', 'cabin', 'embarked']

And we can identify the unique categories for each variables like this:


.. code:: python

    encoder.encoder_dict_

.. code:: python

    {'sex': ['female'],
     'cabin': ['M', 'E', 'C', 'D', 'B', 'A', 'F', 'T'],
     'embarked': ['S', 'C', 'Q']}

We can now encode the categorical variables:

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

And here we see the resulting dataframe:

.. code:: python

          pclass        age  sibsp  parch     fare  sex_female  cabin_M  cabin_E  \
    501        2  13.000000      0      1  19.5000           1        1        0
    588        2   4.000000      1      1  23.0000           1        1        0
    402        2  30.000000      1      0  13.8583           1        1        0
    1193       3  29.881135      0      0   7.7250           0        1        0
    686        3  22.000000      0      0   7.7250           1        1        0

          cabin_C  cabin_D  cabin_B  cabin_A  cabin_F  cabin_T  embarked_S  \
    501         0        0        0        0        0        0           1
    588         0        0        0        0        0        0           1
    402         0        0        0        0        0        0           0
    1193        0        0        0        0        0        0           0
    686         0        0        0        0        0        0           0

          embarked_C  embarked_Q
    501            0           0
    588            0           0
    402            1           0
    1193           0           1
    686            0           1


Encoding variables of type numeric
----------------------------------

By default, Feature-engine's :class:`OneHotEncoder()` will only encode categorical
features. If you attempt to encode a variable of numeric dtype, it will raise an error.
To avoid this error, you can instruct the encoder to ignore the data type format as
follows:


.. code:: python

    enc = OneHotEncoder(
        variables=['pclass'],
        drop_last=True,
        ignore_format=True,
        )

    enc.fit(X_train)

    train_t = enc.transform(X_train)
    test_t = enc.transform(X_test)

    print(train_t.head())

Note that pclass had numeric values instead of strings, and it was one hot encoded by
the transformer into 2 dummies:

.. code:: python

             sex        age  sibsp  parch     fare cabin embarked  pclass_2  \
    501   female  13.000000      0      1  19.5000     M        S         1
    588   female   4.000000      1      1  23.0000     M        S         1
    402   female  30.000000      1      0  13.8583     M        C         1
    1193    male  29.881135      0      0   7.7250     M        Q         0
    686   female  22.000000      0      0   7.7250     M        Q         0

          pclass_3
    501          0
    588          0
    402          0
    1193         1
    686          1

Encoding binary variables into 1 dummy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Feature-engine's :class:`OneHotEncoder()` we can encode all categorical variables
into k dummies and the binary variables into k-1 by setting the encoder as follows:

.. code:: python

    ohe = OneHotEncoder(
        variables=['sex', 'cabin','embarked'],
        drop_last=False,
        drop_last_binary=True,
        )

    train_t = ohe.fit_transform(X_train)
    test_t = ohe.transform(X_test)

    print(train_t.head())

As we see in the following input, for the variable sex, we have only have 1 dummy,
and for all the rest we have k dummies:

.. code:: python

          pclass        age  sibsp  parch     fare  sex_female  cabin_M  cabin_E  \
    501        2  13.000000      0      1  19.5000           1        1        0
    588        2   4.000000      1      1  23.0000           1        1        0
    402        2  30.000000      1      0  13.8583           1        1        0
    1193       3  29.881135      0      0   7.7250           0        1        0
    686        3  22.000000      0      0   7.7250           1        1        0

          cabin_C  cabin_D  cabin_B  cabin_A  cabin_F  cabin_T  cabin_G  \
    501         0        0        0        0        0        0        0
    588         0        0        0        0        0        0        0
    402         0        0        0        0        0        0        0
    1193        0        0        0        0        0        0        0
    686         0        0        0        0        0        0        0

          embarked_S  embarked_C  embarked_Q  embarked_Missing
    501            1           0           0                 0
    588            1           0           0                 0
    402            0           1           0                 0
    1193           0           0           1                 0


Encoding frequent categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the categorical variables are highly cardinal, we may end up with very big datasets
after one hot encoding. In addition, if some of these variables are fairly constant or
fairly similar, we may end up with one hot encoded features that are highly correlated,
if not identical. To avoid this behaviour, we can encode only the most frequent categories.

To encode the 2 most frequent categories of each categorical column, we set up the
transformer as follows:

.. code:: python

    ohe = OneHotEncoder(
        top_categories=2,
        variables=['pclass', 'cabin', 'embarked'],
        ignore_format=True,
        )

    train_t = ohe.fit_transform(X_train)
    test_t = ohe.transform(X_test)

    print(train_t.head())

As we see in the resulting dataframe, we created only 2 dummies per variable:

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

Finally, if we want to obtain the column names in the resulting dataframe we can do the
following:

.. code:: python

    encoder.get_feature_names_out()

We see the names of the columns below:

.. code:: python

    ['sex',
     'age',
     'sibsp',
     'parch',
     'fare',
     'pclass_3',
     'pclass_1',
     'cabin_M',
     'cabin_C',
     'embarked_S',
     'embarked_C']

Considerations
--------------

Encoding categorical variables into k dummies, will handle unknown categories automatically.
Those features not seen during training will show zeroes in all dummies.

Encoding categorical features into k-1 dummies, will cause unseen data to be treated as
the category that is dropped.

Encoding the top categories will make unseen categories part of the group of less popular
categories.

If you add a big number of dummy variables to your data, many may be identical or highly
correlated. Consider dropping identical and correlated features with the transformers
from the :ref:`selection module <selection_user_guide>`.

For alternative encoding methods used in data science check the :class:`OrdinalEncoder()`
and other encoders included in the :ref:`encoding module <encoding_user_guide>`.


Tutorials, books and courses
----------------------------

For more details into :class:`OneHotEncoder()`'s functionality visit:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/OneHotEncoder.ipynb>`_

For tutorials about this and other data preprocessing methods check out our online course:

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