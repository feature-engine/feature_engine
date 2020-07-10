WoERatioCategoricalEncoder
==========================

The WoERatioCategoricalEncoder() replaces the labels by the weight of evidence or the ratio of
probabilities. It only works for binary classification.

The weight of evidence is given by: np.log( p(1) / p(0) )
    
The target probability ratio is given by: p(1) / p(0)

The CountFrequencyCategoricalEncoder() works only with categorical variables. A list of variables can
be indicated, or the encoder will automatically select all categorical variables in the train set.

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
	rare_encoder = ce.RareLabelCategoricalEncoder(tol=0.03, n_categories=2,
    					variables=['cabin', 'pclass', 'embarked'])

	# fit and transform data
	train_t = rare_encoder.fit_transform(X_train)
	test_t = rare_encoder.transform(X_train)

	# set up a weight of evidence encoder
	woe_encoder = ce.WoERatioCategoricalEncoder(
    	encoding_method='woe', variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	woe_encoder.fit(train_t, y_train)

	# transform
	train_t = woe_encoder.transform(train_t)
	test_t = woe_encoder.transform(test_t)

	woe_encoder.encoder_dict_


.. code:: python

    {'cabin': {'B': 1.6299623810120747,
    'C': 0.7217038208351837,
    'D': 1.405081209799324,
    'E': 1.405081209799324,
    'Rare': 0.7387452866900354,
    'n': -0.35752781962490193},
    'pclass': {1: 0.9453018143294478,
    2: 0.21009172435857942,
    3: -0.5841726684724614},
    'embarked': {'C': 0.6999054533737715,
    'Q': -0.05044494288988759,
    'S': -0.20113381737960143}}

API Reference
-------------

.. autoclass:: feature_engine.categorical_encoders.WoERatioCategoricalEncoder
    :members: