RareLabelCategoricalEncoder
===========================
The RareLabelCategoricalEncoder() groups infrequent categories altogether into one new category called
'Rare' or a different string indicated by the user. We need to specify the minimum percentage of observations
a category should show to be preserved and the minimum number of unique categories a variable should
have to be re-grouped.

The RareLabelCategoricalEncoder() works only with categorical variables. A list of variables can
be indicated, or the encoder will automatically select all categorical variables in the train set.

.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine import categorical_encoders as ce

    def load_titanic():
        data = pd.read_csv(
            'https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
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

    # set up the encoder
    encoder = ce.RareLabelCategoricalEncoder(tol=0.03, n_categories=2,
                                             variables=['cabin', 'pclass', 'embarked'],
                                             replace_with='Rare', return_object=True)

    # fit the encoder
    encoder.fit(X_train)

    # transform the data
    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    encoder.encoder_dict_


.. code:: python

	{'cabin': Index(['n', 'C', 'B', 'E', 'D'], dtype='object'),
	 'pclass': array([2, 3, 1], dtype='int64'),
	 'embarked': array(['S', 'C', 'Q'], dtype=object)}


API Reference
-------------

.. autoclass:: feature_engine.categorical_encoders.RareLabelCategoricalEncoder
    :members: