OrdinalEncoder
==============
The OrdinalEncoder() replaces the categories by digits, starting from 0 to k-1, where k is the
number of different categories. If we select "arbitrary", then the encoder will assign numbers as the labels
appear in the variable (first come first served). If we select "ordered", the encoder will assign numbers
following the mean of the target value for that label. So labels for which the mean of the target is higher
will get the number 0, and those where the mean of the target is smallest will get the number k-1.

The OrdinalEncoder() works only with categorical variables. A list of variables can
be indicated, or the encoder will automatically select all categorical variables in the train set.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import OrdinalEncoder

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
	encoder = OrdinalEncoder(encoding_method='ordered', variables=['pclass', 'cabin', 'embarked'])

	# fit the encoder
	encoder.fit(X_train, y_train)

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)

	encoder.encoder_dict_


.. code:: python

	{'pclass': {3: 0, 2: 1, 1: 2},
	 'cabin': {'T': 0,
	  'n': 1,
	  'G': 2,
	  'A': 3,
	  'C': 4,
	  'F': 5,
	  'D': 6,
	  'E': 7,
	  'B': 8},
	 'embarked': {'S': 0, 'Q': 1, 'C': 2}}


API Reference
-------------

.. autoclass:: feature_engine.encoding.OrdinalEncoder
    :members: