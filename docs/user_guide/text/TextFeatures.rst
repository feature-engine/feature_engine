.. _text_features:

.. currentmodule:: feature_engine.text

Extracting Features from Text
=============================

Short pieces of text are often found among the variables in our datasets. For example, in insurance, a text variable can describe the circumstances of an accident. Customer feedback is also stored as a text variable. 

While text data as such can't be used to train machine learning models, we can extract a lot of numerical information from these texts, which can provide predictive features to train machine learning models. 

Feature-engine allows you to quickly extract numerical features from short pieces of text, to complement your predictive models. These features aim to capture a piece of text’s complexity by looking at some statistical parameters of the text, such as the word length and count, the number of words and unique words used, the number of sentences, and so on. :class:`TextFeatures()` extracts many numerical features from text out-of-the-box.

TextFeatures
------------

:class:`TextFeatures()` extracts numerical features from text/string variables.
This transformer is useful for extracting basic text statistics that can be used
as features in machine learning models. Users must explicitly specify which columns
contain text data via the ``variables`` parameter.

Unlike scikit-learn's CountVectorizer or TfidfVectorizer which create sparse matrices,
:class:`TextFeatures()` extracts metadata features that remain in DataFrame format
and can be easily combined with other Feature-engine or sklearn transformers in a pipeline.

Text Features
-------------

:class:`TextFeatures()` can extract the following features from a text piece:

- **char_count**: Number of characters in the text
- **word_count**: Number of words (whitespace-separated tokens)
- **sentence_count**: Number of sentences (based on .!? punctuation)
- **avg_word_length**: Average length of words
- **digit_count**: Number of digit characters
- **letter_count**: Number of alphabetic characters (a-z, A-Z)
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
- **lexical_diversity**: Ratio of unique words to total words

The **number of sentences** is inferred by :class:`TextFeatures()` by counting blocks of
sentence-ending punctuation (., !, ?) as a proxy for sentence boundaries. This means that
multiple consecutive punctuation marks (e.g., "!!!" or "??") are counted as a single
sentence-ending, which avoids overestimating the count in emphatic text.

However, this is still a simple heuristic. It won't handle edge cases like abbreviations
(e.g., 'Dr.', 'U.S.', 'e.g.', 'i.e.') or text without punctuation. These abbreviations
will be counted as sentence endings, resulting in an overestimate of the actual sentence count.

The features **number of unique words** and **lexical diversity** are intended to capture the complexity of the text. Simpler texts have few unique words and tend to repeat them. More complex texts use a wider array of words and tend not to repeat them. Hence, in more complex texts, both the number of unique words and the lexical diversity are greater.

Handling missing values
-----------------------

By default, :class:`TextFeatures()` raises an error if the variables contain missing values.
This behavior can be changed by setting the parameter ``missing_values`` to ``'ignore'``.
In this case, missing values will be treated as empty strings, and the numerical features
will be calculated accordingly (e.g., word count and character count will be 0) as shown in the following example:

.. code:: python

    import pandas as pd
    import numpy as np
    from feature_engine.text import TextFeatures

    # Create sample data with NaN
    X = pd.DataFrame({
        'text': ['Hello', np.nan, 'World']
    })

    # Set up the transformer to ignore missing values
    tf = TextFeatures(
        variables=['text'],
        features=['char_count'],
        missing_values='ignore'
    )

    # Transform
    X_transformed = tf.fit_transform(X)

    print(X_transformed)

In the resulting dataframe, we see that the row with NaN returned 0 in the character count:

.. code-block:: none

    text  text_char_count
    0  Hello                5
    1    NaN                0
    2  World                5

Python demo
-----------

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

    print(X)

The input dataframe looks like this:

.. code-block:: none

                                           review          title
    0  This product is AMAZING! Best purchase ever.  Great Product
    1             Not great. Would not recommend.   Disappointed
    2       OK for the price. 3 out of 5 stars.        Average
    3                     TERRIBLE!!! DO NOT BUY!          Awful

Now let's extract 5 specific text features: the number of words, the number of characters, the number of sentences, whether the text has digits, and the ratio of upper- to lowercase:

.. code:: python

    # Set up the transformer with specific features
    tf = TextFeatures(
        variables=['review'],
        features=['word_count', 'char_count', 'sentence_count', 'has_digits', 'uppercase_ratio']
    )

    # Fit and transform
    tf.fit(X)
    X_transformed = tf.transform(X)

    print(X_transformed)

In the following output, we see the resulting dataframe containing the numerical features extracted from the pieces of text:

.. code-block:: none

       review                                         title          review_word_count  review_char_count
    0  This product is AMAZING! Best purchase ever.    Great Product                  7                 45
    1  Not great. Would not recommend.                Disappointed                   5                 31
    2  OK for the price. 3 out of 5 stars.            Average                        8                 35
    3  TERRIBLE!!! DO NOT BUY!                        Awful                          4                 23

       review_sentence_count  review_has_digits  review_uppercase_ratio
    0                      2                  0                0.066667
    1                      2                  0                0.032258
    2                      2                  3                0.057143
    3                      2                  0                0.608696

Extracting all features
~~~~~~~~~~~~~~~~~~~~~~~

By default, if no text features are specified, all available features will be extracted:

.. code:: python

    # Extract all features from a single text column
    tf = TextFeatures(variables=['review'])
    tf.fit(X)
    X_transformed = tf.transform(X)

    print(X_transformed.head())

The output dataframe contains all 20 text features extracted from the ``review`` column:

.. code-block:: none

       review_char_count  review_word_count  review_sentence_count  review_avg_word_length
    0                 45                  7                      2                5.428571
    1                 31                  5                      2                5.200000
    2                 35                  8                      2                3.375000
    3                 23                  4                      2                4.750000

       review_digit_count  review_letter_count  review_uppercase_count  review_lowercase_count
    0                   0                   38                       3                      35
    1                   0                   26                       1                      25
    2                   2                   25                       2                      23
    3                   0                   14                      14                       0

       review_special_char_count  review_whitespace_count  review_whitespace_ratio
    0                          1                        6                 0.133333
    1                          2                        4                 0.129032
    2                          2                        7                 0.200000
    3                          3                        3                 0.130435

       review_digit_ratio  review_uppercase_ratio  review_has_digits  review_has_uppercase
    0            0.000000                0.066667                  0                     1
    1            0.000000                0.032258                  0                     1
    2            0.057143                0.057143                  1                     1
    3            0.000000                0.608696                  0                     1

       review_is_empty  review_starts_with_uppercase  review_ends_with_punctuation
    0                0                             1                             1
    1                0                             1                             1
    2                0                             1                             1
    3                0                             1                             1

       review_unique_word_count  review_lexical_diversity
    0                         7                  1.000000
    1                         4                  0.800000
    2                         8                  1.000000
    3                         4                  1.000000
Dropping original columns
~~~~~~~~~~~~~~~~~~~~~~~~~

You can drop the original text columns after extracting features, by setting the parameter ``drop_original`` to ``True``:

.. code:: python

    tf = TextFeatures(
        variables=['review'],
        features=['word_count', 'char_count'],
        drop_original=True
    )

    tf.fit(X)
    X_transformed = tf.transform(X)

    print(X_transformed)

The original ``review`` column has been removed, and only the ``title`` column and the extracted features remain:

.. code-block:: none

              title  review_word_count  review_char_count
    0  Great Product                  7                 45
    1  Disappointed                   5                 31
    2       Average                   8                 35
    3         Awful                   4                 23

Combining with scikit-learn Bag-of-Words
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most NLP tasks, it is common to use bag-of-words (e.g., ``CountVectorizer``) or TF-IDF (e.g., ``TfidfVectorizer``) to represent the text. :class:`TextFeatures()` can be used alongside these methods to provide additional metadata that might improve model performance.

In the following example, we compare a baseline model using only TF-IDF with a model that combines TF-IDF and :class:`TextFeatures()` metadata:

.. code:: python

    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split

    from feature_engine.text import TextFeatures

    # Load and split data
    data = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.hockey'])
    df = pd.DataFrame({'text': data.data, 'target': data.target})
    X_train, X_test, y_train, y_test = train_test_split(
        df[['text']], df['target'], test_size=0.3, random_state=42
    )

    print(X_train.head())

The input dataframe contains the raw text of newsgroup posts:

.. code-block:: none

                                                     text
    562  From: xxx@yyy.zzz (John Smith)\nSubject: Re:...
    459  From: aaa@bbb.ccc (Jane Doe)\nSubject: Shutt...
    21   From: ddd@eee.fff\nSubject: Space Station Fr...
    892  From: ggg@hhh.iii\nSubject: NHL Scores\nOrga...
    317  From: jjj@kkk.lll (Bob Wilson)\nSubject: Re:...

Now let's set up two pipelines to compare a baseline model using only TF-IDF with a model that combines TF-IDF and :class:`TextFeatures()` metadata:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # 1. Baseline: TF-IDF only
    tfidf_pipe = Pipeline([
        ('vec', ColumnTransformer([
            ('tfidf', TfidfVectorizer(max_features=500), 'text')
        ])),
        ('clf', LogisticRegression())
    ])
    tfidf_pipe.fit(X_train, y_train)
    print(f"TF-IDF Accuracy: {tfidf_pipe.score(X_test, y_test):.3f}")

    # 2. Combined: TextFeatures + TF-IDF
    combined_pipe = Pipeline([
        ('features', ColumnTransformer([
            ('text_meta', TextFeatures(variables=['text']), 'text'),
            ('tfidf', TfidfVectorizer(max_features=500), 'text')
        ])),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    combined_pipe.fit(X_train, y_train)
    print(f"Combined Accuracy: {combined_pipe.score(X_test, y_test):.3f}")

Below we see the accuracy of a model trained using only the bag of words, respect to a model trained using both the bag of words and the additional meta data:

.. code-block:: none

    TF-IDF Accuracy: 0.957
    Combined Accuracy: 0.963

By adding statistical metadata through :class:`TextFeatures()`, we provided the model with information about text length, complexity, and style that is not explicitly captured by a word-count-based approach like TF-IDF, leading to a small but noticeable improvement in performance.
