.. _match_categories:

.. currentmodule:: feature_engine.preprocessing

MatchCategories
===============

:class:`MatchCategories()` ensures that categorical variables are encoded as pandas
'categorical' dtype, instead of generic python 'object' or other dtypes.

Under the hood, 'categorical' dtype is a representation that maps each
category to an integer, thus providing a more memory-efficient object
structure than e.g., 'str', and allowing faster grouping, mapping, and similar
operations on the resulting object.

MatchCategories() remembers the encodings or levels that represent each
category, and can thus can be used to ensure that the correct encoding gets
applied when passing categorical data to modeling packages that support this
dtype, or to prevent unseen categories from reaching a further transformer
or estimator in a pipeline, for example.

Let's explore this with an example. First we load the Titanic dataset and split it into
a train and a test set:

.. code:: python

    import numpy as np
    import pandas as pd

    from feature_engine.preprocessing import MatchCategories


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

Now, we set up :class:`MatchCategories()` and fit it to the train set.

.. code:: python

    # set up the transformer
    match_categories = MatchCategories(errors="ignore")

    # learn the mapping of categories to integers in the train set
    match_categories.fit(train)

:class:`MatchCategories()` stores the mappings from the train set in its attribute:

.. code:: python

    # the transformer stores the mappings for categorical variables
    match_categories.category_dict_

.. code:: python

    {'pclass': Int64Index([1, 2, 3], dtype='int64'),
     'sex': Index(['female', 'male'], dtype='object'),
     'cabin': Index(['A', 'B', 'C', 'D', 'E', 'F', 'T', 'n'], dtype='object'),
     'embarked': Index(['C', 'Q', 'S'], dtype='object')}


If we transform the test dataframe using the same `match_categories` object,
categorical variables will be converted to a 'category' dtype with the same
numeration (mapping from categories to integers) that was applied to the train
dataset:

.. code:: python

    # encoding that would be gotten from the train set
    train.embarked.unique()

.. code:: python

    array(['S', 'C', 'Q'], dtype=object)

.. code:: python
    
    # encoding that would be gotten from the test set
    test.embarked.unique()

.. code:: python

    array(['Q', 'S', 'C'], dtype=object)

.. code:: python
    
    # with 'match_categories', the encoding remains the same
    match_categories.transform(train).embarked.cat.categories

.. code:: python

    Index(['C', 'Q', 'S'], dtype='object')

.. code:: python

    # this will have the same encoding as the train set
    match_categories.transform(test).embarked.cat.categories

.. code:: python

    Index(['C', 'Q', 'S'], dtype='object')



If some category was not present in the training data, it will not map
to any integer and will this not get encoded (behavior here depends on what one
passed for 'errors'):

.. code:: python

    # categories present in the train data
    train.cabin.unique()

.. code:: python

    array(['B', 'C', 'E', 'D', 'A', 'n', 'T', 'F'], dtype=object)

.. code:: python
    
    # categories present in the test data - 'G' is new
    test.cabin.unique()

.. code:: python

    array(['n', 'F', 'E', 'G'], dtype=object)

.. code:: python

    match_categories.transform(train).cabin.unique()

.. code:: python

    ['B', 'C', 'E', 'D', 'A', 'n', 'T', 'F']
    Categories (8, object): ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'n']

.. code:: python
    
    # unseen category 'G' will not get mapped to any integer
    match_categories.transform(test).cabin.unique()

.. code:: python

    ['n', 'F', 'E', NaN]
    Categories (8, object): ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'n']


When to use the transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^


This transformer is useful when creating custom transformers for categorical columns,
as well as when passing categorical columns to modeling packages that support them
natively but which leaving their encoding to the user.
