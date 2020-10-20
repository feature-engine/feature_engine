OneHotEncoder
=============

The OneHotEncoder() replaces categorical variables by a set of binary variables,
one per unique category. The encoder has the option to create k or k-1 binary variables,
where k is the number of unique categories.

The encoder can also create binary variables for the n most popular categories, n being
determined by the user. This means, if we encode the 6 more popular categories, we will
only create binary variables for those categories, and the rest will be dropped.

The OneHotEncoder() works only with categorical variables. A list of variables can
be indicated, or the encoder will automatically select all categorical variables in the
train set.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import OneHotEncoder

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
	encoder = OneHotEncoder( top_categories=2, variables=['pclass', 'cabin', 'embarked'], drop_last=False)

	# fit the encoder
	encoder.fit(X_train)

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)

	encoder.encoder_dict_


.. code:: python

	{'pclass': [3, 1], 'cabin': ['n', 'C'], 'embarked': ['S', 'C']}



API Reference
-------------

.. autoclass:: feature_engine.encoding.OneHotEncoder
    :members: