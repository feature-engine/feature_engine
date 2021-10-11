MeanEncoder
===========

API Reference
-------------

.. autoclass:: feature_engine.encoding.MeanEncoder
    :members:

Example
-------

The MeanEncoder() replaces categories with the mean of the target per category. For
example, if we are trying to predict default rate, and our data has the variable city,
with categories, London, Manchester and Bristol, and the default rate per city is 0.1,
0.5, and 0.3, respectively, the encoder will replace London by 0.1, Manchester by 0.5
and Bristol by 0.3.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import MeanEncoder

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

	# set up the encoder
	encoder = MeanEncoder(variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	encoder.fit(X_train, y_train)

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)

	encoder.encoder_dict_


.. code:: python

	{'cabin': {'A': 0.5294117647058824,
	  'B': 0.7619047619047619,
	  'C': 0.5633802816901409,
	  'D': 0.71875,
	  'E': 0.71875,
	  'F': 0.6666666666666666,
	  'G': 0.5,
	  'T': 0.0,
	  'n': 0.30484330484330485},
	 'pclass': {1: 0.6173913043478261,
	  2: 0.43617021276595747,
	  3: 0.25903614457831325},
	 'embarked': {'C': 0.5580110497237569,
	  'Q': 0.37349397590361444,
	  'S': 0.3389570552147239}}


