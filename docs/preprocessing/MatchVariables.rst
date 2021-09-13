MatchVariables
==============

API Reference
-------------

.. autoclass:: feature_engine.preprocessing.MatchVariables
    :members:


Example
-------

MatchVariables() ensures that the columns in the test set are identical to those
in the train set.

If the test set contains additional columns, they are dropped. Alternatively, if the
test set lacks columns that were present in the train set, they will be added with a
value determined by the user, for example np.nan.


.. code:: python

    import numpy as np
    import pandas as pd

    from feature_engine.preprocessing import MatchVariables


    # Load dataset
    def load_titanic():
        data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
        data = data.replace('?', np.nan)
        data['cabin'] = data['cabin'].astype(str).str[0]
        data['pclass'] = data['pclass'].astype('O')
        data['age'] = data['age'].astype('float')
        data['fare'] = data['fare'].astype('float')
        data['embarked'].fillna('C', inplace=True)
        data.drop(
            labels=['name', 'ticket', 'boat', 'body', 'home.dest'],
            axis=1, inplace=True,
        )
        return data

    # load data as pandas dataframe
    data = load_titanic()

    # Split test and train
    train = data.iloc[0:1000, :]
    test = data.iloc[1000:, :]

    # set up the transformer
    match_cols = MatchVariables(missing_values="ignore")

    # learn the variables in the train set
    match_cols.fit(train)

    # the transformer stores the input variables
    match_cols.input_features_


.. code:: python

    ['pclass',
     'survived',
     'sex',
     'age',
     'sibsp',
     'parch',
     'fare',
     'cabin',
     'embarked']


.. code:: python

    # Let's drop some columns in the test set for the demo
    test_t = test.drop(["sex", "age"], axis=1)

    test_t.head()

.. code:: python

         pclass  survived  sibsp  parch     fare cabin embarked
    1000      3         1      0      0   7.7500     n        Q
    1001      3         1      2      0  23.2500     n        Q
    1002      3         1      2      0  23.2500     n        Q
    1003      3         1      2      0  23.2500     n        Q
    1004      3         1      0      0   7.7875     n        Q


.. code:: python

    # the transformer adds the columns back
    test_tt = match_cols.transform(test_t)

    test_tt.head()

.. code:: python

    The following variables are added to the DataFrame: ['sex', 'age']

         pclass  survived  sex  age  sibsp  parch     fare cabin embarked
    1000      3         1  NaN  NaN      0      0   7.7500     n        Q
    1001      3         1  NaN  NaN      2      0  23.2500     n        Q
    1002      3         1  NaN  NaN      2      0  23.2500     n        Q
    1003      3         1  NaN  NaN      2      0  23.2500     n        Q
    1004      3         1  NaN  NaN      0      0   7.7875     n        Q



Note how the missing columns were added back to the transformed test set, with
missing values, in the position (i.e., order) in which they were in the train set.

Similarly, if the test set contained additional columns, those would be removed:

.. code:: python

    # let's add some columns for the demo
    test_t[['var_a', 'var_b']] = 0

    test_t.head()

.. code:: python

         pclass  survived  sibsp  parch     fare cabin embarked  var_a  var_b
    1000      3         1      0      0   7.7500     n        Q      0      0
    1001      3         1      2      0  23.2500     n        Q      0      0
    1002      3         1      2      0  23.2500     n        Q      0      0
    1003      3         1      2      0  23.2500     n        Q      0      0
    1004      3         1      0      0   7.7875     n        Q      0      0


.. code:: python

    test_tt = match_cols.transform(test_t)

    test_tt.head()

.. code:: python

    The following variables are added to the DataFrame: ['age', 'sex']
    The following variables are dropped from the DataFrame: ['var_a', 'var_b']

         pclass  survived  sex  age  sibsp  parch     fare cabin embarked
    1000      3         1  NaN  NaN      0      0   7.7500     n        Q
    1001      3         1  NaN  NaN      2      0  23.2500     n        Q
    1002      3         1  NaN  NaN      2      0  23.2500     n        Q
    1003      3         1  NaN  NaN      2      0  23.2500     n        Q
    1004      3         1  NaN  NaN      0      0   7.7875     n        Q


Now, the transformer simultaneously added the missing columns with NA as values and
removed the additional columns from the resulting dataset.

These transformer is useful in "predict then optimize type of problems". In such cases,
a machine learning model is trained on a certain dataset, with certain input features.
Then, test sets are "post-processed" according to scenarios that want to be modelled.
For example, "what would have happened if the customer received an email campaign"?
where the variable "receive_campaign" would be turned from 0 -> 1.

While creating these modelling datasets, a lot of meta data e.g., "scenario number",
"time scenario was generated", etc, could be added to the data. Then we need to pass
these data over to the model to obtain the modelled prediction.

MatchVariables() provides an easy an elegant way to remove the additional metadeta,
while returning datasets with the input features in the correct order, allowing the
different scenarios to be modelled directly inside a machine learning pipeline.