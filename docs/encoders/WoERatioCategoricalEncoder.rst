WoERatioCategoricalEncoder
==========================

The WoERatioCategoricalEncoder() replaces the labels by the weight of evidence or the ratio of
probabilities. It only works for binary classification.

The weight of evidence is given by: np.log( p(1) / p(0) )
    
The target probability ratio is given by: p(1) / p(0)

The CountFrequencyCategoricalEncoder() works only with categorical variables. A list of variables can
be indiacated, or the imputer will automatically select all categorical variables in the train set.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine import categorical_encoders as ce

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
	rare_encoder = ce.RareLabelCategoricalEncoder(tol=0.03, n_categories=5,
    					variables=['cabin', 'pclass', 'embarked'])

	# fit and transform data
	train_t = rare_encoder.fit_transform(X_train)
	test_t = rare_encoder.transform(X_train)

	# set up a weight of evidence encoder
	encoder = ce.WoERatioCategoricalEncoder(
    	encoding_method='woe', variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	encoder.fit(train_t, y_train)

	# transform
	train_t = rare_encoder.transform(train_t)
	test_t = rare_encoder.transform(test_t)

	encoder.encoder_dict_


.. code:: python

	{'cabin': {'B': 1.1631508098056806,
	  'C': 0.2548922496287902,
	  'D': 0.9382696385929302,
	  'E': 0.9382696385929302,
	  'Rare': 0.2719337154836416,
	  'n': -0.8243393908312957},
	 'pclass': {1: 0.4784902431230542,
	  2: -0.25671984684781396,
	  3: -1.0509842396788551},
	 'embarked': {'C': 0.23309388216737797,
	  'Q': -0.5172565140962812,
	  'S': -0.6679453885859952}}

API Reference
-------------

.. autoclass:: feature_engine.categorical_encoders.WoERatioCategoricalEncoder
    :members: