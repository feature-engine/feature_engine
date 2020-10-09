PRatioEncoder
=============

The PRatioEncoder() replaces the labels by the ratio of probabilities. It only works
for binary classification.
    
The target probability ratio is given by: p(1) / p(0)

The log of the target probability ratio is: np.log( p(1) / p(0) )

The PRatioEncoder() works only with categorical variables. A list of variables can
be indicated, or the encoder will automatically select all categorical variables in the
train set.

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

	# set up a rare label encoder
	rare_encoder = RareLabelEncoder(tol=0.03, n_categories=2, variables=['cabin', 'pclass', 'embarked'])

	# fit and transform data
	train_t = rare_encoder.fit_transform(X_train)
	test_t = rare_encoder.transform(X_train)

	# set up a weight of evidence encoder
	pratio_encoder = PRatioEncoder(encoding_method='ratio', variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	pratio_encoder.fit(train_t, y_train)

	# transform
	train_t = pratio_encoder.transform(train_t)
	test_t = pratio_encoder.transform(test_t)

	pratio_encoder.encoder_dict_


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

API Reference
-------------

.. autoclass:: feature_engine.encoding.PRatioEncoder
    :members: