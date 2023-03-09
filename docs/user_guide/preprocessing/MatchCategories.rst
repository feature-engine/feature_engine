.. _match_categories:

.. currentmodule:: feature_engine.preprocessing

MatchCategories
===============

:class:`MatchCategories()` ensures that categorical variables are encoded as pandas
'categorical' dtype instead of generic python 'object' or other dtypes.

Under the hood, 'categorical' dtype is a representation that maps each
category to an integer, thus providing a more memory-efficient object
structure than, for example, 'str', and allowing faster grouping, mapping, and similar
operations on the resulting object.

:class:`MatchCategories()` remembers the encodings or levels that represent each
category, and can thus can be used to ensure that the correct encoding gets
applied when passing categorical data to modeling packages that support this
dtype, or to prevent unseen categories from reaching a further transformer
or estimator in a pipeline, for example.

Let's explore this with an example. First we load the Titanic dataset and split it into
a train and a test sets:

.. code:: python

    from feature_engine.preprocessing import MatchCategories
    from feature_engine.datasets import load_titanic

    # Load dataset
    data = load_titanic(
        predictors_only=True,
        handle_missing=True,
        cabin="letter_only",
    )

    data['pclass'] = data['pclass'].astype('O')

    # Split test and train
    train = data.iloc[0:1000, :]
    test = data.iloc[1000:, :]

Now, we set up :class:`MatchCategories()` and fit it to the train set.

.. code:: python

    # set up the transformer
    match_categories = MatchCategories(missing_values="ignore")

    # learn the mapping of categories to integers in the train set
    match_categories.fit(train)

:class:`MatchCategories()` stores the mappings from the train set in its attribute:

.. code:: python

    # the transformer stores the mappings for categorical variables
    match_categories.category_dict_

.. code:: python

    {'pclass': Int64Index([1, 2, 3], dtype='int64'),
     'sex': Index(['female', 'male'], dtype='object'),
     'cabin': Index(['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T'], dtype='object'),
     'embarked': Index(['C', 'Missing', 'Q', 'S'], dtype='object')}

If we transform the test dataframe using the same `match_categories` object,
categorical variables will be converted to a 'category' dtype with the same
numeration (mapping from categories to integers) that was applied to the train
dataset:

.. code:: python

    # encoding that would be gotten from the train set
    train.embarked.unique()

.. code:: python

    array(['S', 'C', 'Missing', 'Q'], dtype=object)

.. code:: python
    
    # encoding that would be gotten from the test set
    test.embarked.unique()

.. code:: python

    array(['Q', 'S', 'C'], dtype=object)

.. code:: python
    
    # with 'match_categories', the encoding remains the same
    match_categories.transform(train).embarked.cat.categories

.. code:: python

    Index(['C', 'Missing', 'Q', 'S'], dtype='object')

.. code:: python

    # this will have the same encoding as the train set
    match_categories.transform(test).embarked.cat.categories

.. code:: python

    Index(['C', 'Missing', 'Q', 'S'], dtype='object')

If some category was not present in the training data, it will not mapped
to any integer and will thus not get encoded. This behavior can be modified through the
parameter `errors`:

.. code:: python

    # categories present in the train data
    train.cabin.unique()

.. code:: python

    array(['B', 'C', 'E', 'D', 'A', 'M', 'T', 'F'], dtype=object)

.. code:: python
    
    # categories present in the test data - 'G' is new
    test.cabin.unique()

.. code:: python

    array(['M', 'F', 'E', 'G'], dtype=object)

.. code:: python

    match_categories.transform(train).cabin.unique()

.. code:: python

    ['B', 'C', 'E', 'D', 'A', 'M', 'T', 'F']
    Categories (8, object): ['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T']

.. code:: python
    
    # unseen category 'G' will not get mapped to any integer
    match_categories.transform(test).cabin.unique()

.. code:: python

    ['M', 'F', 'E', NaN]
    Categories (8, object): ['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T']


When to use the transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This transformer is useful when creating custom transformers for categorical columns,
as well as when passing categorical columns to modeling packages which support them
natively but leave the variable casting to the user, such as ``lightgbm`` or ``glum``.
