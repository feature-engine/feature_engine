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

    # Load dataset
    data = load_titanic(handle_missing=True)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['embarked'].fillna('C', inplace=True)

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                data.drop(['survived', 'name', 'ticket'], axis=1),
                data['survived'], test_size=0.3, random_state=0)

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

    # transform the data
    train_t = transformer.transform(X_train)
    train_t.head()

         pclass     sex        age  sibsp     fare     boat  \
    501       2  female  13.000000      0  19.5000       14   
    588       2  female   4.000000      1  23.0000       14   
    402       2  female  30.000000      1  13.8583       12   
    1193      3    male  29.881135      0   7.7250  Missing   
    686       3  female  22.000000      0   7.7250       13   

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
