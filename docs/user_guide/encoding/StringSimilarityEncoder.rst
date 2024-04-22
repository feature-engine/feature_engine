.. _string_similarity:

.. currentmodule:: feature_engine.encoding


StringSimilarityEncoder
=======================

The :class:`StringSimilarityEncoder()` replaces categorical variables with a set of float
variables that capture the similarity between the category names. The new variables
have values between 0 and 1, where 0 indicates no similarity and 1 is an exact
match between the names of the categories.

To calculate the similarity between the categories, :class:`StringSimilarityEncoder()`
uses Gestalt pattern matching. Under the hood, :class:`StringSimilarityEncoder()` uses
the `quick_ratio` method from the `SequanceMatcher()` from `difflib`.

The similarity is calculated as:

.. math::

    GPM = 2 M / T

where T is the total number of elements in both sequences and M is the number of matches.

For example, the similarity between the categories "dog" and "dig" is 0.66. T is the
total number of elements in both categories, that is 6. There are 2 matches between
the words, the letters d and g, so: 2 * M / T = 2 * 2 / 6 = 0.66.

Output of the :class:`StringSimilarityEncoder()`
------------------------------------------------

Let's create a dataframe with the categories "dog", "dig" and "cat":

.. code:: python

    import pandas as pd
    from feature_engine.encoding import StringSimilarityEncoder

    df = pd.DataFrame({"words": ["dog", "dig", "cat"]})
    df

We see the dataframe in the following output:

.. code:: python

      words
    0   dog
    1   dig
    2   cat

Let's now encode the variable:

.. code:: python

    encoder =  StringSimilarityEncoder()
    dft = encoder.fit_transform(df)
    dft

We see the encoded variables below:


.. code:: python

       words_dog  words_dig  words_cat
    0   1.000000   0.666667        0.0
    1   0.666667   1.000000        0.0
    2   0.000000   0.000000        1.0


Note that :class:`StringSimilarityEncoder()` replaces the original variables by the
distance variables.

:class:`StringSimilarityEncoder()` vs One-hot encoding
------------------------------------------------------

String similarity encoding is similar to one-hot encoding, in the sense that each category is
encoded as a new variable. But the values, instead of 1 or 0, are the similarity
between the observation's category and the dummy variable. It is suitable for poorly
defined (or 'dirty') categorical variables.

Encoding only popular categories
--------------------------------

The :class:`StringSimilarityEncoder()` can also create similarity variables for the *n* most popular
categories, *n* being determined by the user. For example, if we encode only the 6 more popular categories, by
setting the parameter `top_categories=6`, the transformer will add variables only
for the 6 most frequent categories. The most frequent categories are those with the largest
number of observations. This behaviour is useful when the categorical variables are highly cardinal,
to control the expansion of the feature space.

Specifying how :class:`StringSimilarityEncoder()` should deal with missing values
---------------------------------------------------------------------------------

The :class:`StringSimilarityEncoder()` has three options for dealing with missing values, which can be
specified with the parameter `missing_values`:

  1. Ignore NaNs (option `ignore`) - will leave the NaN in the resulting dataframe after transformation.
     Could be useful, if the next step in the pipeline is imputation or if the machine learning algorithm
     can handle missing data out-of-the-box.
  2. Impute NaNs (option `impute`) - will impute NaN with an empty string, and then calculate the similarity
     between the empty string and the variable's categories. Most of the time, the similarity value will be
     0 in resulting dataframe. This is the default option.
  3. Raise an error (option `raise`) - will raise an error if NaN is present during `fit`, `transform` or
     `fit_transform`. Could be useful for debugging and monitoring purposes.


Important
---------

:class:`StringSimilarityEncoder()` will encode unseen categories by out-of-the-box, by measuring the
string similarity to the seen categories.

No text preprocessing is applied by :class:`StringSimilarityEncoder()`. Be mindful of preparing
string categorical variables if needed.

:class:`StringSimilarityEncoder()` works with categorical variables by default. And it has the option to
encode numerical variables as well. This is useful, when the values of the numerical variables are more
useful as strings, than as numbers. For example, for variables like barcode.

Examples
--------

Let's look at an example using the Titanic Dataset. First we load the data and divide it
into a train and a test set:

.. code:: python

    import string
    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import StringSimilarityEncoder

    def clean_titanic():
        translate_table = str.maketrans('' , '', string.punctuation)
        data = load_titanic()
        data['home.dest'] = (
        data['home.dest']
        .str.strip()
        .str.translate(translate_table)
        .str.replace('  ', ' ')
        .str.lower()
        )
        data['name'] = (
        data['name']
        .str.strip()
        .str.translate(translate_table)
        .str.replace('  ', ' ')
        .str.lower()
        )
        data['ticket'] = (
        data['ticket']
        .str.strip()
        .str.translate(translate_table)
        .str.replace('  ', ' ')
        .str.lower()
        )
        return data

    data = clean_titanic()
    # Separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['survived', 'sex', 'cabin', 'embarked'], axis=1),
        data['survived'],
        test_size=0.3,
        random_state=0
    )

    X_train.head()

Below, we see the first rows of the dataset:

.. code:: python

        pclass                             name  age  sibsp  parch  \
    501        2  mellinger miss madeleine violet   13      0      1   
    588        2                  wells miss joan    4      1      1   
    402        2     duran y more miss florentina   30      1      0   
    1193       3                 scanlan mr james  NaN      0      0   
    686        3       bradley miss bridget delia   22      0      0   

                ticket     fare boat body  \
    501         250644     19.5   14  NaN   
    588          29103       23   14  NaN   
    402   scparis 2148  13.8583   12  NaN   
    1193         36209    7.725  NaN  NaN   
    686         334914    7.725   13  NaN   

                                                home.dest  
    501                             england bennington vt  
    588                                 cornwall akron oh  
    402                       barcelona spain havana cuba  
    1193                                              NaN  
    686   kingwilliamstown co cork ireland glens falls ny 


Now, we set up the encoder to encode only the 2 most frequent categories of each of the
3 indicated categorical variables:

.. code:: python

    # set up the encoder
    encoder = StringSimilarityEncoder(
        top_categories=2,
        variables=['name', 'home.dest', 'ticket'],
        ignore_format=True
        )

    # fit the encoder
    encoder.fit(X_train)

With `fit()` the encoder will learn the most popular categories of the variables, which
are stored in the attribute `encoder_dict_`.

.. code:: python

	encoder.encoder_dict_

.. code:: python

    {
      'name': ['mellinger miss madeleine violet', 'barbara mrs catherine david'],
      'home.dest': ['', 'new york ny'],
      'ticket': ['ca 2343', 'ca 2144']
    }

The `encoder_dict_` contains the categories that will derive similarity variables for each
categorical variable.

With transform, we go ahead and encode the variables. Note that the
:class:`StringSimilarityEncoder()` will drop the original variables.

.. code:: python

    # transform the data
    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    test_t.head()

Below, we see the resulting dataframe:

.. code:: python

        pclass  age  sibsp  parch    fare boat body  \
    1139       3   38      0      0  7.8958  NaN  NaN   
    533        2   21      0      1      21   12  NaN   
    459        2   42      1      0      27  NaN  NaN   
    1150       3  NaN      0      0    14.5  NaN  NaN   
    393        2   25      0      0    31.5  NaN  NaN   

        name_mellinger miss madeleine violet  name_barbara mrs catherine david  \
    1139                              0.454545                          0.550000   
    533                               0.615385                          0.524590   
    459                               0.596491                          0.603774   
    1150                              0.641509                          0.693878   
    393                               0.408163                          0.666667   

        home.dest_nan  home.dest_new york ny  ticket_ca 2343  ticket_ca 2144  
    1139            1.0               0.000000        0.461538        0.461538  
    533             0.0               0.370370        0.307692        0.307692  
    459             0.0               0.352941        0.461538        0.461538  
    1150            1.0               0.000000        0.307692        0.307692  
    393             0.0               0.437500        0.666667        0.666667


More details
------------

For more details into :class:`StringSimilarityEncoder()`'s functionality visit:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/StringSimilarityEncoder.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
