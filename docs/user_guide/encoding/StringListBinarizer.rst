.. _string_list_binarizer:

.. currentmodule:: feature_engine.encoding

StringListBinarizer
===================

:class:`StringListBinarizer()` replaces categorical variables containing lists of strings
or comma-delimited strings with a set of binary variables (dummy variables) representing
each one of the unique tags or categories present across all observations.

This transformer is particularly useful for handling multi-label categorical columns
where each row might have multiple values, such as ``"action, comedy"`` or
``"romance, thriller, action"``. The transformer splits these strings by a specified separator,
collects all unique tags, and then applies one-hot encoding on them. It can also natively
handle columns structured as Python lists, like ``["action", "comedy"]``.

Python example
--------------

Let's look at an example. We generate a toy dataset with multi-label genre information
stored as comma-delimited strings:

.. code:: python

    import pandas as pd
    from feature_engine.encoding import StringListBinarizer

    X = pd.DataFrame(dict(
        user_id = [1, 2, 3],
        genres = ["action, comedy", "comedy", "action, thriller"]
    ))

    print(X)

.. code:: python

       user_id            genres
    0        1    action, comedy
    1        2            comedy
    2        3  action, thriller

Now, we set up the :class:`StringListBinarizer()`. Since our strings are separated by a
comma and a space, we specify ``separator=", "``.

.. code:: python

    slb = StringListBinarizer(
        variables=["genres"],
        separator=", "
    )

    slb.fit(X)

During `fit`, the enoder splits the strings, identifies the unique categories across
the entire dataset, and saves them in its `encoder_dict_` attribute.

.. code:: python

    print(slb.encoder_dict_)
    # {'genres': ['action', 'comedy', 'thriller']}

We can now use `transform` to get the dummy variables. The original column is dropped by default.

.. code:: python

    X_encoded = slb.transform(X)
    print(X_encoded)

.. code:: python

       user_id  genres_action  genres_comedy  genres_thriller
    0        1              1              1               0
    1        2              0              1               0
    2        3              1              0               1

As we see, each row now has a 1 in the columns corresponding to the genres it originally contained,
and 0 otherwise. Unseen categories encountered during transform will simply be ignored (i.e. all
dummy columns will be 0 for those extra components).
