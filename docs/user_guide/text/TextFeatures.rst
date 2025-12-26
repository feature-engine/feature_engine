.. _text_features:

.. currentmodule:: feature_engine.text

TextFeatures
============

The :class:`TextFeatures()` extracts numerical features from text/string variables.
This transformer is useful for extracting basic text statistics that can be used
as features in machine learning models.

Unlike scikit-learn's CountVectorizer or TfidfVectorizer which create sparse matrices,
:class:`TextFeatures()` extracts metadata features that remain in DataFrame format
and can be easily combined with other Feature-engine transformers in a pipeline.

Available Features
~~~~~~~~~~~~~~~~~~

The transformer can extract the following features:

- **char_count**: Number of characters in the text
- **word_count**: Number of words (whitespace-separated tokens)
- **sentence_count**: Number of sentences (based on .!? punctuation)
- **avg_word_length**: Average length of words
- **digit_count**: Number of digit characters
- **uppercase_count**: Number of uppercase letters
- **lowercase_count**: Number of lowercase letters
- **special_char_count**: Number of special characters (non-alphanumeric)
- **whitespace_count**: Number of whitespace characters
- **whitespace_ratio**: Ratio of whitespace to total characters
- **digit_ratio**: Ratio of digits to total characters
- **uppercase_ratio**: Ratio of uppercase to total characters
- **has_digits**: Binary indicator if text contains digits
- **has_uppercase**: Binary indicator if text contains uppercase
- **is_empty**: Binary indicator if text is empty
- **starts_with_uppercase**: Binary indicator if text starts with uppercase
- **ends_with_punctuation**: Binary indicator if text ends with .!?
- **unique_word_count**: Number of unique words (case-insensitive)
- **unique_word_ratio**: Ratio of unique words to total words

Example
~~~~~~~

Let's create a dataframe with text data and extract features:

.. code:: python

    import pandas as pd
    from feature_engine.text import TextFeatures

    # Create sample data
    X = pd.DataFrame({
        'review': [
            'This product is AMAZING! Best purchase ever.',
            'Not great. Would not recommend.',
            'OK for the price. 3 out of 5 stars.',
            'TERRIBLE!!! DO NOT BUY!',
        ],
        'title': [
            'Great Product',
            'Disappointed',
            'Average',
            'Awful',
        ]
    })

Now let's extract specific text features:

.. code:: python

    # Set up the transformer with specific features
    tf = TextFeatures(
        variables=['review'],
        features=['word_count', 'char_count', 'has_digits', 'uppercase_ratio']
    )

    # Fit and transform
    tf.fit(X)
    X_transformed = tf.transform(X)

    print(X_transformed.columns.tolist())

Output:

.. code:: python

    ['review', 'title', 'review_word_count', 'review_char_count',
     'review_has_digits', 'review_uppercase_ratio']

Extracting all features
~~~~~~~~~~~~~~~~~~~~~~~

By default, if no features are specified, all available features will be extracted:

.. code:: python

    # Extract all features from all text columns
    tf = TextFeatures()
    tf.fit(X)
    X_transformed = tf.transform(X)

    # This will create 19 new columns for each text variable
    print(f"Original columns: {len(X.columns)}")
    print(f"Transformed columns: {len(X_transformed.columns)}")

Dropping original columns
~~~~~~~~~~~~~~~~~~~~~~~~~

You can drop the original text columns after extracting features:

.. code:: python

    tf = TextFeatures(
        variables=['review'],
        features=['word_count', 'char_count'],
        drop_original=True
    )

    tf.fit(X)
    X_transformed = tf.transform(X)

    # 'review' column is now removed
    print(X_transformed.columns.tolist())

Using in a Pipeline
~~~~~~~~~~~~~~~~~~~

:class:`TextFeatures()` works seamlessly with scikit-learn pipelines:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Create a pipeline
    pipe = Pipeline([
        ('text_features', TextFeatures(
            variables=['review'],
            features=['word_count', 'char_count', 'uppercase_ratio'],
            drop_original=True
        )),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

API Reference
-------------

.. autoclass:: TextFeatures
    :members:
    :inherited-members:
