.. _string_similarity:

.. currentmodule:: feature_engine.encoding


StringSimilarityEncoder
=============

The :class:`StringSimilarityEncoder()` replaces categorical variables by a set of
float variables representing similarity between unique categories in the variable.
This new variables will have values in range between 0 and 1, where 0 is the least similar
and 1 is the complete match.
This encoding is an alternative to OneHotEncoder in the case of poorly
defined (or 'dirty') categorical variables.

For example, from the categorical variable "Profession" with categories
('Data Analyst', 'Business Analyst', 'Product Analyst', 'Project Manager', 'Software Engineer'),
we can generate the float variables which will take value from 0 to 1, based on how text strings are similar.

**Encoding only popular categories**

The encoder can also create similarity variables for the n most popular categories, n being
determined by the user. For example, if we encode only the 6 more popular categories, by
setting the parameter `top_categories=6`, the transformer will add variables only
for the 6 most frequent categories. The most frequent categories are those with the biggest
number of observations. The remaining categories will not be encoded. This behaviour is useful
when the categorical variables are highly cardinal, to control the expansion of the feature space.

**Specifying how encoder should deal with missing values**

The encoder has three options on dealing with missing values, specified by parameter `handle_missing`:
  1. Ignore NaNs (option `ignore`) - will leave NaN in resulting dataframe after transformation.
     Could be useful, if the next step in the pipeline will be imputer or ML algortih with imputing capabilities.
  2. Impute NaNs (option `impute`) - will impute NaN with an empty string, most of the time it will
     be represented as 0 in resulting dataframe. This is the default option.
  3. Raise an error (option `error`) - will raise an error if NaN is present during `fit`, `transform` or
     `fit_transform` method. Could be useful for debugging and monitoring purposes.

**Note**

This encoder will encode new categories by measuring string similarity between seen unseen categories.

No preprocessing is applied, so it is on user to prepare string categorical variables for this transformer.

Let's look at an example using the Titanic Dataset. First we load the data and divide it
into a train and a test set:

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import StringSimilarityEncoder

	# Load dataset
	def load_titanic():
		data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
		data = data.replace('?', np.nan)
		data['home.dest'] = data['home.dest'].str.strip().str.replace(',', '').str.replace('/', '').str.replace('  ', ' ')
    data['name'] = data['name'].str.strip().str.replace(',', '').str.replace('.', '', regex=False).str.replace('  ', ' ')
    data['ticket'] = data['ticket'].str.strip().str.replace('/', '').str.replace('.', '', regex=False).str.replace('  ', ' ')
		return data
	
	data = load_titanic()

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
				data.drop(['survived', 'sex', 'cabin', 'embarked'], axis=1),
				data['survived'], test_size=0.3, random_state=0)

Now, we set up the encoder to encode only the 2 most frequent categories of each of the
3 indicated categorical variables:

.. code:: python

	# set up the encoder
	encoder = StringSimilarityEncoder(top_categories=2, variables=['name', 'home.dest', 'ticket'])

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

With transform, we go ahead and encode the variables. Note that by default, the
:class:`StringSimilarityEncoder()` will drop the original variables.

.. code:: python

	# transform the data
	train_t = encoder.transform(X_train)
	test_t = encoder.transform(X_test)

More details
^^^^^^^^^^^^

For more details into :class:`StringSimilarityEncoder()`'s functionality visit:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/StringSimilarityEncoder.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
