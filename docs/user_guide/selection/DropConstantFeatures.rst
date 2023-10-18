.. _drop_constant:

.. currentmodule:: feature_engine.selection

DropConstantFeatures
====================


The :class:`DropConstantFeatures()` drops constant and quasi-constant variables from a dataframe.
By default, it drops only constant variables. Constant variables have a single
value. Quasi-constant variables have a single value in most of its observations.

This transformer works with numerical and categorical variables, and it offers a pretty straightforward
way of reducing the feature space. Be mindful though, that depending on the context, quasi-constant
variables could be useful.

**Example**

Let's see how to use :class:`DropConstantFeatures()` in an example with the Titanic dataset. We
first load the data and separate it into train and test:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.selection import DropConstantFeatures

    X, y = load_titanic(
        return_X_y_frame=True,
        handle_missing=True,
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )

Now, we set up the :class:`DropConstantFeatures()` to remove features that show the same
value in more than 70% of the observations:

.. code:: python

    # set up the transformer
    transformer = DropConstantFeatures(tol=0.7)


With `fit()` the transformer finds the variables to drop:

.. code:: python

    # fit the transformer
    transformer.fit(X_train)

The variables to drop are stored in the attribute `features_to_drop_`:

.. code:: python

    transformer.features_to_drop_

.. code:: python

    ['parch', 'cabin', 'embarked', 'body']


We see in the following code snippets that for the variables parch and embarked, more
than 70% of the observations displayed the same value:

.. code:: python

    X_train['embarked'].value_counts(normalize = True)

.. code:: python

    S          0.711790
    C          0.195415
    Q          0.090611
    Missing    0.002183
    Name: embarked, dtype: float64


71% of the passengers embarked in S.

.. code:: python

    X_train['parch'].value_counts(normalize = True)

.. code:: python

    0    0.771834
    1    0.125546
    2    0.086245
    3    0.005459
    4    0.004367
    5    0.003275
    6    0.002183
    9    0.001092
    Name: parch, dtype: float64

77% of the passengers had 0 parent or child. Because of this, these features were
deemed constant and removed.

With `transform()`, we can go ahead and drop the variables from the data:

.. code:: python

    train_t = transformer.transform(X_train)
    test_t = transformer.transform(X_test)

    print(train_t.head())

We see the resulting dataframe below:

.. code:: python

          pclass                               name     sex        age  sibsp  \
    501        2  Mellinger, Miss. Madeleine Violet  female  13.000000      0
    588        2                  Wells, Miss. Joan  female   4.000000      1
    402        2     Duran y More, Miss. Florentina  female  30.000000      1
    1193       3                 Scanlan, Mr. James    male  29.881135      0
    686        3       Bradley, Miss. Bridget Delia  female  22.000000      0

                 ticket     fare     boat  \
    501          250644  19.5000       14
    588           29103  23.0000       14
    402   SC/PARIS 2148  13.8583       12
    1193          36209   7.7250  Missing
    686          334914   7.7250       13

                                                  home.dest
    501                            England / Bennington, VT
    588                                Cornwall / Akron, OH
    402                     Barcelona, Spain / Havana, Cuba
    1193                                            Missing
    686   Kingwilliamstown, Co Cork, Ireland Glens Falls...


More details
^^^^^^^^^^^^

In this Kaggle kernel we use :class:`DropConstantFeatures()` together with other feature selection algorithms:

- `Kaggle kernel <https://www.kaggle.com/solegalli/feature-selection-with-feature-engine>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.

For more details about this and other feature selection methods check out these resources:

- `Feature selection for machine learning <https://www.trainindata.com/p/feature-selection-for-machine-learning>` _, online course.
- `Feature selection in machine learning <https://leanpub.com/feature-selection-in-machine-learning>` _, book.
