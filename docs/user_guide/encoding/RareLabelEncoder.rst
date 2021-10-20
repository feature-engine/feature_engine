.. _rarelabel_encoder:

.. currentmodule:: feature_engine.encoding

RareLabelEncoder
================

The :class:`RareLabelEncoder()` groups infrequent categories into one new category
called 'Rare' or a different string indicated by the user. We need to specify the
minimum percentage of observations a category should have to be preserved and the
minimum number of unique categories a variable should have to be re-grouped.

**tol**

In the parameter `tol` we indicate the minimum proportion of observations a category
should have, not to be grouped. In other words, categories which frequency, or proportion
of observations is <= `tol` will be grouped into a unique term.

**n_categories**

In the parameter `n_categories` we indicate the minimum cardinality of the categorical
variable in order to group infrequent categories. For example, if `n_categories=5`,
categories will be grouped only in those categorical variables with more than 5 unique
categories. The rest will be ignored.

This parameter is useful when we have big datasets and do not have time to examine all
categorical variables individually. This way, we ensure that variables with low cardinality
are not reduced any further.

**max_n_categories**

In the parameter `max_n_categories` we indicate the maximum number of unique categories
that we want in the encoded variable. If `max_n_categories=5`, then the most popular 5
categories will remain in the variable after the encoding, all other will be grouped into
a single category.

This parameter is useful if we are going to perform one hot encoding at the back of it,
to control the expansion of the feature space.

**Example**

Let's look at an example using the Titanic Dataset.


.. code:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    from feature_engine.encoding import RareLabelEncoder

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
    encoder = RareLabelEncoder(tol=0.03, n_categories=2, variables=['cabin', 'pclass', 'embarked'],
                               replace_with='Rare')

    # fit the encoder
    encoder.fit(X_train)

    # transform the data
    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    encoder.encoder_dict_

In the `encoder_dict_` we find the most frequent categories per variable to encode.
Any category that is not in this dictionary, will be grouped.

.. code:: python

	{'cabin': Index(['n', 'C', 'B', 'E', 'D'], dtype='object'),
	 'pclass': array([2, 3, 1], dtype='int64'),
	 'embarked': array(['S', 'C', 'Q'], dtype=object)}

You can also specify the maximum number of categories that can be considered frequent
using the `max_n_categories` parameter.

.. code:: python

    from feature_engine.encoding import RareLabelEncoder
    import pandas as pd
    data = {'var_A': ['A'] * 10 + ['B'] * 10 + ['C'] * 2 + ['D'] * 1}
    data = pd.DataFrame(data)
    data['var_A'].value_counts()

.. code:: python

    A    10
    B    10
    C     2
    D     1
    Name: var_A, dtype: int64

.. code:: python

    rare_encoder = RareLabelEncoder(tol=0.05, n_categories=3)
    rare_encoder.fit_transform(data)['var_A'].value_counts()

.. code:: python

    A       10
    B       10
    C        2
    Rare     1
    Name: var_A, dtype: int64

.. code:: python

    rare_encoder = RareLabelEncoder(tol=0.05, n_categories=3, max_n_categories=2)
    Xt = rare_encoder.fit_transform(data)
    Xt['var_A'].value_counts()

.. code:: python

    A       10
    B       10
    Rare     3
    Name: var_A, dtype: int64


More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/RareLabelEncoder.ipynb>`_
