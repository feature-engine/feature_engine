DecisionTreeCategoricalEncoder
==============================
The DecisionTreeCategoricalEncoder() replaces categories in the variable with 
the predictions of a decision tree. The transformer first encodes categorical
variables into numerical variables using ordinal encoding. You have the option
to have the integers assigned to the categories as they appear in the variable, 
or ordered by the mean value of the target per category. After this, the transformer
fits with this numerical variable a decision tree to predict the target variable.
Finally, the original categorical variable is replaced by the predictions of
the decision tree.

The DecisionTreeCategoricalEncoder() works only with categorical variables. A
list of variables can be indicated, or alternatively, the imputer will automatically
select all categorical variables in the train set.

Note that a decision tree is fit per every single variable. With this transformer
variables are not combined.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine import categorical_encoders as ce

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

.. code:: python

    # set up the encoder
    encoder = ce.DecisionTreeCategoricalEncoder(variables=['cabin', 'pclass', 'embarked'],
              random_state=0)

    # fit the encoder
    encoder.fit(X_train, y_train)

    # transform the data
    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    train_t[['cabin', 'pclass', 'embarked']].head(10)

.. code:: python

         cabin    pclass  embarked
    501   0.304843  0.307580  0.338957
    588   0.304843  0.307580  0.338957
    402   0.304843  0.307580  0.558011
    1193  0.304843  0.307580  0.373494
    686   0.304843  0.307580  0.373494
    971   0.304843  0.307580  0.373494
    117   0.649533  0.617391  0.558011
    540   0.304843  0.307580  0.338957
    294   0.649533  0.617391  0.558011
    261   0.649533  0.617391  0.338957


API Reference
-------------

.. autoclass:: feature_engine.categorical_encoders.DecisionTreeCategoricalEncoder
    :members:
