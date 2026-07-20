.. _match_variables:

.. currentmodule:: feature_engine.preprocessing

MatchVariables
==============

:class:`MatchVariables()` ensures that the columns in the test set are identical to those
in the train set.

If the test set contains additional columns, they are dropped. Alternatively, if the
test set lacks columns that were present in the train set, they will be added with a
value determined by the user, for example np.nan. :class:`MatchVariables()` will also
return the variables in the order seen in the train set.

Let's explore this with an example. We start with the imports:

.. code:: python

    from feature_engine.preprocessing import MatchVariables
    from feature_engine.datasets import load_titanic

Next, we load the Titanic dataset:

.. code:: python

    # Load dataset
    data = load_titanic(
        predictors_only=True,
        cabin="letter_only",
    )

    data['pclass'] = data['pclass'].astype('O')

And we split it into a train set and a test set:

.. code:: python

    # Split test and train
    train = data.iloc[0:1000, :]
    test = data.iloc[1000:, :]

Now, we set up :class:`MatchVariables()` and fit it to the train set:

.. code:: python

    # set up the transformer
    match_cols = MatchVariables(missing_values="ignore")

    # learn the variables in the train set
    match_cols.fit(train)

:class:`MatchVariables()` stores the variables from the train set in its attribute:

.. code:: python

    # the transformer stores the input variables
    match_cols.feature_names_in_

These are the variables in the train set, in the order in which they appear:

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

Now, we drop some columns in the test set, to simulate a test set that is missing
variables that were present in the train set:

.. code:: python

    # Let's drop some columns in the test set for the demo
    test_t = test.drop(["sex", "age"], axis=1)

    test_t.head()

We see that `sex` and `age` are no longer in the dataframe:

.. code:: python

         pclass  survived  sibsp  parch     fare cabin embarked
    1000      3         1      0      0   7.7500     n        Q
    1001      3         1      2      0  23.2500     n        Q
    1002      3         1      2      0  23.2500     n        Q
    1003      3         1      2      0  23.2500     n        Q
    1004      3         1      0      0   7.7875     n        Q

If we transform the dataframe with the dropped columns using :class:`MatchVariables()`,
we see that the new dataframe contains all the variables, and those that were missing
are now back in the data, with np.nan values as default:

.. code:: python

    # the transformer adds the columns back
    test_tt = match_cols.transform(test_t)

    test_tt.head()

Indeed, `sex` and `age` are back, filled with missing values:

.. code:: python

    The following variables are added to the DataFrame: ['age', 'sex']
         pclass  survived  sex  age  sibsp  parch     fare cabin embarked
    1000      3         1  NaN  NaN      0      0   7.7500     n        Q
    1001      3         1  NaN  NaN      2      0  23.2500     n        Q
    1002      3         1  NaN  NaN      2      0  23.2500     n        Q
    1003      3         1  NaN  NaN      2      0  23.2500     n        Q
    1004      3         1  NaN  NaN      0      0   7.7875     n        Q

Note how the missing columns were added back to the transformed test set, with
missing values, in the position (i.e., order) in which they were in the train set.

Similarly, if the test set contained additional columns, those would be removed. To
test that, let's add some extra columns to the test set:

.. code:: python

    # let's add some columns for the demo
    test_t[['var_a', 'var_b']] = 0

    test_t.head()

We now have 2 extra columns, `var_a` and `var_b`, that were not present in the train set:

.. code:: python

         pclass  survived  sibsp  parch     fare cabin embarked  var_a  var_b
    1000      3         1      0      0   7.7500     n        Q      0      0
    1001      3         1      2      0  23.2500     n        Q      0      0
    1002      3         1      2      0  23.2500     n        Q      0      0
    1003      3         1      2      0  23.2500     n        Q      0      0
    1004      3         1      0      0   7.7875     n        Q      0      0

And now, we transform the data with :class:`MatchVariables()`:

.. code:: python

    test_tt = match_cols.transform(test_t)

    test_tt.head()

The transformer simultaneously added the missing columns with NA as values and removed
the additional columns from the resulting dataset:

.. code:: python

    The following variables are added to the DataFrame: ['age', 'sex']
    The following variables are dropped from the DataFrame: ['var_b', 'var_a']
         pclass  survived  sex  age  sibsp  parch     fare cabin embarked
    1000      3         1  NaN  NaN      0      0   7.7500     n        Q
    1001      3         1  NaN  NaN      2      0  23.2500     n        Q
    1002      3         1  NaN  NaN      2      0  23.2500     n        Q
    1003      3         1  NaN  NaN      2      0  23.2500     n        Q
    1004      3         1  NaN  NaN      0      0   7.7875     n        Q

However, if we look closely, the dtypes for the `sex` variable do not match. This could
cause issues if other transformations depend upon having the correct dtypes. This is the
dtype in the train set:

.. code:: python

    train.sex.dtype

Which is:

.. code:: python

    dtype('O')

And this is the dtype in the transformed test set:

.. code:: python

    test_tt.sex.dtype

Which does not match:

.. code:: python

    dtype('float64')

Set the `match_dtypes` parameter to `True` in order to align the dtypes as well:

.. code:: python

    match_cols_and_dtypes = MatchVariables(missing_values="ignore", match_dtypes=True)
    match_cols_and_dtypes.fit(train)

    test_ttt = match_cols_and_dtypes.transform(test_t)

We see in the messages that the `sex` dtype was changed to match that of the train set:

.. code:: python

    The following variables are added to the DataFrame: ['sex', 'age']
    The following variables are dropped from the DataFrame: ['var_b', 'var_a']
    The sex dtype is changing from  float64 to object

Now the dtype matches:

.. code:: python

    test_ttt.sex.dtype

Which is:

.. code:: python

    dtype('O')

By default, :class:`MatchVariables()` will print out messages indicating which variables
were added, removed and altered. We can switch off the messages through the parameter `verbose`.


When to use the transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This transformer is useful in "predict then optimise" type of problems. In such cases,
a machine learning model is trained on a certain dataset, with certain input features.
Then, test sets are "post-processed" according to scenarios that need to be modelled.
For example, "what would have happened if the customer received an email campaign?"
Here, the variable "receive_campaign" would be turned from 0 to 1.

While creating these modelling datasets, a lot of metadata, e.g., "scenario number",
"time scenario was generated", etc, could be added to the data. Then we need to pass
these data over to the model to obtain the modelled prediction.

:class:`MatchVariables()` provides an easy and elegant way to remove the additional metadata,
while returning datasets with the input features in the correct order, allowing the
different scenarios to be modelled directly inside a machine learning pipeline.

More details
^^^^^^^^^^^^

You can also find a similar implementation of the example shown in this page in the
following Jupyter notebook:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/preprocessing/MatchVariables.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
