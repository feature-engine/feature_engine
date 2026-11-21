.. _match_categories:

.. currentmodule:: feature_engine.preprocessing

MatchCategories
===============

:class:`MatchCategories()` ensures that categorical variables are encoded as pandas
'categorical' dtype, or polars 'Enum' dtype, instead of generic python 'object',
string or other dtypes.

Under the hood, 'categorical' dtype is a representation that maps each
category to an integer, thus providing a more memory-efficient object
structure than, for example, 'str', and allowing faster grouping, mapping, and similar
operations on the resulting object.

:class:`MatchCategories()` remembers the encodings or levels that represent each
category, and can thus be used to ensure that the correct encoding gets
applied when passing categorical data to modelling packages that support this
dtype, or to prevent unseen categories from reaching a further transformer
or estimator in a pipeline, for example.

.. attention::

    **New in version 2.0:** When `variables` is `None`, :class:`MatchCategories()` used to
    raise an error if the dataframe contained no categorical variables. You can now
    set the new parameter `return_empty` to `True` to make the transformer return an
    empty list of variables and skip matching the categories instead, leaving the
    dataframe unchanged. This lets you reuse the same pipeline across different
    datasets or projects, some of which may not contain categorical variables,
    without building a tailored pipeline for each one. `return_empty` will default to
    `True` from version 2.1 onwards.

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

    {'pclass': Index([1, 2, 3], dtype='int64'),
     'sex': Index(['female', 'male'], dtype='str'),
     'cabin': Index(['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T'], dtype='str'),
     'embarked': Index(['C', 'Missing', 'Q', 'S'], dtype='str')}

To see why this matters, let's compare the order in which the categories of `embarked`
appear in the raw train and test sets. This is the order in the train set:

.. code:: python

    train.embarked.unique()

We obtain the following order:

.. code:: python

    <StringArray>
    ['S', 'C', 'Missing', 'Q']
    Length: 4, dtype: str

And this is the order in the test set:

.. code:: python

    test.embarked.unique()

Which is different from the train set:

.. code:: python

    <StringArray>
    ['Q', 'S', 'C']
    Length: 3, dtype: str

The categories appear in a different order in each set. If we transform the dataframes
using the same `match_categories` object, categorical variables will be converted to a
'category' dtype with the same numeration (mapping from categories to integers) that was
applied to the train dataset. This is the order we obtain for the train set:

.. code:: python

    match_categories.transform(train).embarked.cat.categories

Which is:

.. code:: python

    Index(['C', 'Missing', 'Q', 'S'], dtype='str')

And this is the order we now obtain for the test set:

.. code:: python

    match_categories.transform(test).embarked.cat.categories

The 2 sets now show exactly the same category order:

.. code:: python

    Index(['C', 'Missing', 'Q', 'S'], dtype='str')

If some category was not present in the training data, it will not be mapped
to any integer and will become a missing value instead. Let's illustrate this with the `cabin` variable. These are the
categories present in the train set:

.. code:: python

    train.cabin.unique()

We obtain the following categories:

.. code:: python

    <StringArray>
    ['B', 'C', 'E', 'D', 'A', 'M', 'T', 'F']
    Length: 8, dtype: str

And these are the categories present in the test set, which include a category, 'G',
that was not seen during training:

.. code:: python

    test.cabin.unique()

We obtain the following categories, including the unseen 'G':

.. code:: python

    <StringArray>
    ['M', 'F', 'E', 'G']
    Length: 4, dtype: str

After transforming the train set, we obtain the same categories as before, now correctly
typed as 'category' dtype:

.. code:: python

    match_categories.transform(train).cabin.unique()

Which are:

.. code:: python

    ['B', 'C', 'E', 'D', 'A', 'M', 'T', 'F']
    Categories (8, str): ['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T']

But when we transform the test set, the unseen category 'G' is not mapped to any integer,
and becomes a missing value instead:

.. code:: python

    match_categories.transform(test).cabin.unique()

We see that 'G' has been replaced by a missing value:

.. code:: python

    ['M', 'F', 'E', NaN]
    Categories (8, str): ['A', 'B', 'C', 'D', 'E', 'F', 'M', 'T']

Because we set `missing_values="ignore"`, :class:`MatchCategories()` warns us that
missing values were introduced:

.. code:: python

    UserWarning: During the encoding, NaN values were introduced in the feature(s) cabin.

With the default `missing_values="raise"`, :class:`MatchCategories()` raises an error
instead, both when the data contains missing values and when unseen categories would
introduce them.

With polars
^^^^^^^^^^^

:class:`MatchCategories()` also works with polars dataframes. In polars, the variables
are cast to the 'Enum' dtype, which, like pandas 'categorical', holds a fixed list of
categories. Let's create a toy train set and test set:

.. code:: python

    import polars as pl
    from feature_engine.preprocessing import MatchCategories

    train = pl.DataFrame({
        "city": ["London", "Paris", "Madrid", "Paris"],
        "rooms": [2, 3, 1, 3],
    })
    test = pl.DataFrame({
        "city": ["Madrid", "Rome", "London", "Paris"],
        "rooms": [1, 2, 4, 3],
    })

We fit :class:`MatchCategories()` to the train set:

.. code:: python

    match_categories = MatchCategories(missing_values="ignore")
    match_categories.fit(train)

    match_categories.category_dict_

With polars, the categories are stored in lists:

.. code:: python

    {'city': ['London', 'Madrid', 'Paris']}

Now we transform the test set:

.. code:: python

    test_t = match_categories.transform(test)
    test_t

The variable `city` is now an 'Enum', and the unseen category 'Rome' became a missing
value:

.. code:: text

    shape: (4, 2)
    ┌────────┬───────┐
    │ city   ┆ rooms │
    │ ---    ┆ ---   │
    │ enum   ┆ i64   │
    ╞════════╪═══════╡
    │ Madrid ┆ 1     │
    │ null   ┆ 2     │
    │ London ┆ 4     │
    │ Paris  ┆ 3     │
    └────────┴───────┘

We can check the categories in the schema:

.. code:: python

    test_t.schema

The categories are the ones learned from the train set:

.. code:: python

    Schema({'city': Enum(categories=['London', 'Madrid', 'Paris']), 'rooms': Int64})

The polars 'Enum' dtype only takes strings. Hence, if we cast numerical variables by
setting `ignore_format=True`, their values become strings. The categories are sorted
in numerical order:

.. code:: python

    match_categories = MatchCategories(
        variables=["rooms"], ignore_format=True, missing_values="ignore"
    )
    match_categories.fit(train)

    match_categories.category_dict_

We see the categories of `rooms` as strings:

.. code:: python

    {'rooms': ['1', '2', '3']}

And these are the values after the transformation, where the unseen value 4 became a
missing value:

.. code:: python

    match_categories.transform(test)

.. code:: text

    shape: (4, 2)
    ┌────────┬───────┐
    │ city   ┆ rooms │
    │ ---    ┆ ---   │
    │ str    ┆ enum  │
    ╞════════╪═══════╡
    │ Madrid ┆ 1     │
    │ Rome   ┆ 2     │
    │ London ┆ null  │
    │ Paris  ┆ 3     │
    └────────┴───────┘


When to use the transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This transformer is useful when creating custom transformers for categorical columns,
as well as when passing categorical columns to modelling packages which support them
natively but leave the variable casting to the user, such as ``lightgbm`` or ``glum``.
