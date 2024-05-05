.. _decisiontree_encoder:

.. currentmodule:: feature_engine.encoding

DecisionTreeEncoder
===================

Categorical encoding is the process of transforming the strings of categorical features
into numbers. Common procedures replace categories with ordinal numbers, counts or
frequencies, or with the target mean value.

We can also replace the categories with the predictions made by a decision tree based
on that category value.

The process consists in fitting a decision tree using a single feature to predict the
target. The decision tree will try to find a relationship between these variables, if
it exists, and then, we'd use the predictions as mappings to replace the categories.

The advantage of these procedure is that it captures some information about the relationship
between the variables during the encoding. And if there is a relationship between the
categorical feature and the target, the resulting encoded variable would have a monotonic
relationship with the target, which can be useful for linear models.

On the downside, it could cause overfitting and it adds computational complexity to the
pipeline, because we are fitting a tree per feature. If you plan to encode your features
with decision trees, make sure you have appropriate validation strategies, and train the
decision trees with regularization.

DecisionTreeEncoder
-------------------

The :class:`DecisionTreeEncoder()` replaces categories in the variable with
the predictions of a decision tree.

The :class:`DecisionTreeEncoder()`  uses Scikit-learn's decision trees under the hood.
As these models can't handle non-numerical data, The :class:`DecisionTreeEncoder()` first
replaces the categories with ordinal numbers, and then fits the trees.

You have the option to encode the categorical values into integers assigned arbitrarily,
or ordered based of the mean target value per category (for more details check the
:class:`OrdinalEncoder()`, which is used by :class:`DecisionTreeEncoder()`under the hood).
You can regulate this behaviour with the parameter `encoding_method`. As decision trees
are able to pick non-linear relationships, replacing categories by arbitrary numbers
should be enough in practice.

After this, the transformer fits with this numerical variable a decision tree to predict
the target variable. Finally, the original categorical variable is replaced by the
predictions of the decision tree.

In the attribute `encoding_dict_` you'll find the mappings from category to numerical
value. The category is the original value and the numerical value is the prediction
of the decision tree for the category.

The motivation for the :class:`DecisionTreeEncoder()` is to try and create monotonic
relationships between the categorical variables and the target.

Python example
--------------

Let's look at an example using the Titanic Dataset. First, let's load the data and
separate it into train and test:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import DecisionTreeEncoder

    X, y = load_titanic(
        return_X_y_frame=True,
        handle_missing=True,
        predictors_only=True,
        cabin="letter_only",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )

    print(X_train[['cabin', 'pclass', 'embarked']].head(10))

We will encode the following categorical variables:

.. code:: python

        cabin  pclass embarked
    501      M       2        S
    588      M       2        S
    402      M       2        C
    1193     M       3        Q
    686      M       3        Q
    971      M       3        Q
    117      E       1        C
    540      M       2        S
    294      C       1        C
    261      E       1        S

We set up the encoder to encode the variables above with 3 fold cross-validation, using
a grid search to find the optimal depth of the decision tree (this is the default
behaviour of the :class:`DecisionTreeEncoder()`). In this example, we optimize the
tree using the roc-auc metric.

.. code:: python

    encoder = DecisionTreeEncoder(
        variables=['cabin', 'pclass', 'embarked'],
        regression=False,
        scoring='roc_auc',
        cv=3,
        random_state=0,
        ignore_format=True)

    encoder.fit(X_train, y_train)

With `fit()` the :class:`DecisionTreeEncoder()` fits 1 decision tree per variable. The
mappings are stored in the `encoding_dict_`:

.. code:: python

    encoder.encoder_dict_

In the following output we see the values that will be used to replace each category
in each variable:

.. code:: python

    {'cabin': {'M': 0.30484330484330485,
      'E': 0.6116504854368932,
      'C': 0.6116504854368932,
      'D': 0.6981132075471698,
      'B': 0.6981132075471698,
      'A': 0.6981132075471698,
      'F': 0.6981132075471698,
      'T': 0.0,
      'G': 0.5},
     'pclass': {2: 0.43617021276595747,
      3: 0.25903614457831325,
      1: 0.6173913043478261},
     'embarked': {'S': 0.3389570552147239,
      'C': 0.553072625698324,
      'Q': 0.37349397590361444,
      'Missing': 1.0}}

Now we can go ahead and transform the categorical variables into numbers, using the
predictions of these trees:

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    train_t[['cabin', 'pclass', 'embarked']].head(10)

We can see the encoded variables below:

.. code:: python

            cabin    pclass  embarked
    501   0.304843  0.436170  0.338957
    588   0.304843  0.436170  0.338957
    402   0.304843  0.436170  0.553073
    1193  0.304843  0.259036  0.373494
    686   0.304843  0.259036  0.373494
    971   0.304843  0.259036  0.373494
    117   0.611650  0.617391  0.553073
    540   0.304843  0.436170  0.338957
    294   0.611650  0.617391  0.553073
    261   0.611650  0.617391  0.338957


Additional resources
--------------------

In the following notebook, you can find more details into the :class:`DecisionTreeEncoder()`
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/DecisionTreeEncoder.ipynb>`_

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

Our book:

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

Both our book and courses are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.