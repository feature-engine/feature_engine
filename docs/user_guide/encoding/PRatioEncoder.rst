.. _pratio_encoder:

.. currentmodule:: feature_engine.encoding

PRatioEncoder
=============

The :class:`PRatioEncoder()` replaces categories by the ratio of the probability of the
target = 1 and the probability of the target = 0.

The target probability ratio is given by:

.. math::

    p(1) / p(0)

The log of the target probability ratio is:

.. math::

    log( p(1) / p(0) )


For example in the variable colour, if the mean of the target = 1 for blue
is 0.8 and the mean of the target = 0  is 0.2, blue will be replaced by:
0.8 / 0.2 = 4 if ratio is selected, or log(0.8/0.2) = 1.386 if log_ratio
is selected.

**Note**

This categorical encoding is exclusive for binary classification.

Let's look at an example using the Titanic Dataset.

First, let's load the data and separate it into train and test:

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import PRatioEncoder, RareLabelEncoder

	# Load dataset
	def load_titanic():
		data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
		data = data.replace('?', np.nan)
		data['cabin'] = data['cabin'].astype(str).str[0]
		data['pclass'] = data['pclass'].astype('O')
		data['embarked'].fillna('C', inplace=True)
		return data

	data = load_titanic()

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
			data.drop(['survived', 'name', 'ticket'], axis=1),
			data['survived'], test_size=0.3, random_state=0)

Before we encode the variables, I would like to group infrequent categories into one
category, called 'Rare'. For this, I will use the :class:`RareLabelEncoder()` as follows:

.. code:: python

	# set up a rare label encoder
	rare_encoder = RareLabelEncoder(tol=0.03, n_categories=2, variables=['cabin', 'pclass', 'embarked'])

	# fit and transform data
	train_t = rare_encoder.fit_transform(X_train)
	test_t = rare_encoder.transform(X_train)

Now, we set up the :class:`PRatioEncoder()` to replace the categories by the probability
ratio, only in the 3 indicated variables:

.. code:: python

	# set up a weight of evidence encoder
	pratio_encoder = PRatioEncoder(encoding_method='ratio', variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	pratio_encoder.fit(train_t, y_train)

With `fit()` the encoder learns the values to replace each category, which are stored in
its `encoder_dict_` parameter:

.. code:: python

	pratio_encoder.encoder_dict_

In the `encoder_dict_` we find the probability ratio for each category in each
variable to encode. This way, we can map the original value to the new value.

.. code:: python

    {'cabin': {'B': 3.1999999999999993,
     'C': 1.2903225806451615
     'D': 2.5555555555555554,
     'E': 2.5555555555555554,
     'Rare': 1.3124999999999998,
     'n': 0.4385245901639344},
     'pclass': {1: 1.6136363636363635,
      2: 0.7735849056603774,
      3: 0.34959349593495936},
      'embarked': {'C': 1.2625000000000002,
      'Q': 0.5961538461538461,
      'S': 0.5127610208816704}}

Now, we can go ahead and encode the variables:

.. code:: python

	# transform
	train_t = pratio_encoder.transform(train_t)
	test_t = pratio_encoder.transform(test_t)


More details
^^^^^^^^^^^^

In the following notebook, you can find more details into the :class:`PRatioEncoder()`
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/PRatioEncoder.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
