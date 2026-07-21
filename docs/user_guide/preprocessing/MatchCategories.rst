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
category, and can thus be used to ensure that the correct encoding gets
applied when passing categorical data to modelling packages that support this
dtype, or to prevent unseen categories from reaching a further transformer
or estimator in a pipeline, for example.

Let's explore this with an example. We start with the imports:

.. code:: python

    from feature_engine.preprocessing import MatchCategories
    from feature_engine.datasets import load_titanic

Next, we load the Titanic dataset:

.. code:: python

    # Load dataset
    data = load_titanic(
        predictors_only=True,
        handle_missing=True,
        cabin="letter_only",
    )

    data['pclass'] = data['pclass'].astype('O')

And we split it into a train set and a test set:

.. code:: python

    # Split test and train
    train = data.iloc[0:1000, :]
    test = data.iloc[1000:, :]

Now, we set up :class:`MatchCategories()` and fit it to the train set:

.. code:: python

    # set up the transformer
    match_categories = MatchCategories(missing_values="ignore")

    # learn the mapping of categories to integers in the train set
    match_categories.fit(train)

:class:`MatchCategories()` stores the mappings from the train set in its attribute:

.. code:: python

    # the transformer stores the mappings for categorical variables
    match_categories.category_dict_

Here are the mappings learnt for each categorical variable:

.. code:: python

    {'pclass': Int64Index([1, 2, 3], dtype='int64'),
     'sex': Index(['female', 'male'], dtype='object'),
     'cabin': Index(['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T'], dtype='object'),
     'embarked': Index(['C', 'Missing', 'Q', 'S'], dtype='object')}

.. note::

    **New in version 2.0:** When `variables` is `None`, :class:`MatchCategories()` used to
    raise an error if the dataframe contained no categorical variables. You can now
    set the new parameter `return_empty` to `True` to make the transformer return an
    empty list of variables and skip matching the categories instead, leaving the
    dataframe unchanged. This lets you reuse the same pipeline across different
    datasets or projects, some of which may not contain categorical variables,
    without building a tailored pipeline for each one. `return_empty` will default to
    `True` from version 2.1 onwards.

To see why this matters, let's compare the order in which the categories of `embarked`
appear in the raw train and test sets. This is the order in the train set:

.. code:: python

    train.embarked.unique()

We obtain the following order:

.. code:: python

    array(['S', 'C', 'Missing', 'Q'], dtype=object)

And this is the order in the test set:

.. code:: python

    test.embarked.unique()

Which is different from the train set:

.. code:: python

    array(['Q', 'S', 'C'], dtype=object)

The categories appear in a different order in each set. If we transform the dataframes
using the same `match_categories` object, categorical variables will be converted to a
'category' dtype with the same numeration (mapping from categories to integers) that was
applied to the train dataset. This is the order we obtain for the train set:

.. code:: python

    match_categories.transform(train).embarked.cat.categories

Which is:

.. code:: python

    Index(['C', 'Missing', 'Q', 'S'], dtype='object')

And this is the order we now obtain for the test set:

.. code:: python

    match_categories.transform(test).embarked.cat.categories

The 2 sets now show exactly the same category order:

.. code:: python

    Index(['C', 'Missing', 'Q', 'S'], dtype='object')

If some category was not present in the training data, it will not be mapped
to any integer and will thus not get encoded. This behaviour can be modified through the
parameter `errors`. Let's illustrate this with the `cabin` variable. These are the
categories present in the train set:

.. code:: python

    train.cabin.unique()

We obtain the following categories:

.. code:: python

    array(['B', 'C', 'E', 'D', 'A', 'M', 'T', 'F'], dtype=object)

And these are the categories present in the test set, which include a category, 'G',
that was not seen during training:

.. code:: python

    test.cabin.unique()

We obtain the following categories, including the unseen 'G':

.. code:: python

    array(['M', 'F', 'E', 'G'], dtype=object)

After transforming the train set, we obtain the same categories as before, now correctly
typed as 'category' dtype:

.. code:: python

    match_categories.transform(train).cabin.unique()

Which are:

.. code:: python

    ['B', 'C', 'E', 'D', 'A', 'M', 'T', 'F']
    Categories (8, object): ['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T']

But when we transform the test set, the unseen category 'G' is not mapped to any integer,
and becomes a missing value instead:

.. code:: python

    match_categories.transform(test).cabin.unique()

We see that 'G' has been replaced by a missing value:

.. code:: python

    ['M', 'F', 'E', NaN]
    Categories (8, object): ['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T']


When to use the transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This transformer is useful when creating custom transformers for categorical columns,
as well as when passing categorical columns to modelling packages which support them
natively but leave the variable casting to the user, such as ``lightgbm`` or ``glum``.
