.. _decisiontree_encoder:

.. currentmodule:: feature_engine.encoding

DecisionTreeEncoder
===================

The :class:`DecisionTreeEncoder()` replaces categories in the variable with
the predictions of a decision tree.

The transformer first encodes categorical variables into numerical variables using
:class:`OrdinalEncoder`. You have the option to have the integers assigned to the
categories as they appear in the variable, or ordered by the mean value of the target
per category. You can regulate this behaviour with the parameter `encoding_method`. As
decision trees are able to pick non-linear relationships, replacing categories by
arbitrary numbers should be enough in practice.

After this, the transformer fits with this numerical variable a decision tree to predict
the target variable. Finally, the original categorical variable is replaced by the
predictions of the decision tree.

The motivation of the :class:`DecisionTreeEncoder()` is to try and create monotonic
relationships between the categorical variables and the target.

Let's look at an example using the Titanic Dataset.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine.encoding import DecisionTreeEncoder

    # Load dataset
    def load_titanic():
            data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
            data = data.replace('?', np.nan)
            data['cabin'] = data['cabin'].astype(str).str[0]
            data['pclass'] = data['pclass'].astype('O')
            data['embarked'].fillna('C', inplace=True)
            return data

    data = load_titanic()

    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                    data.drop(['survived', 'name', 'ticket'], axis=1),
                    data['survived'], test_size=0.3, random_state=0)

    X_train[['cabin', 'pclass', 'embarked']].head(10)

We will encode the following categorical variables:

.. code:: python

         cabin pclass embarked
   501      n      2        S
   588      n      2        S
   402      n      2        C
   1193     n      3        Q
   686      n      3        Q
   971      n      3        Q
   117      E      1        C
   540      n      2        S
   294      C      1        C
   261      E      1        S

We set up the encoder and encode the variables:

.. code:: python

    # set up the encoder
    encoder = DecisionTreeEncoder(
         variables=['cabin', 'pclass', 'embarked'],
         regression=False,
         scoring='roc_auc',
         cv=3,
         random_state=0)

    # fit the encoder
    encoder.fit(X_train, y_train)

    # transform the data
    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    train_t[['cabin', 'pclass', 'embarked']].head(10)

We can see the encoded variables below:

.. code:: python

             cabin    pclass  embarked
    501   0.304843  0.436170  0.338957
    588   0.304843  0.436170  0.338957
    402   0.304843  0.436170  0.558011
    1193  0.304843  0.259036  0.373494
    686   0.304843  0.259036  0.373494
    971   0.304843  0.259036  0.373494
    117   0.611650  0.617391  0.558011
    540   0.304843  0.436170  0.338957
    294   0.611650  0.617391  0.558011
    261   0.611650  0.617391  0.338957


More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/DecisionTreeEncoder.ipynb>`_

