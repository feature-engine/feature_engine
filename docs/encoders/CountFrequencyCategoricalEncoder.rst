CountFrequencyCategoricalEncoder
================================
The CountFrequencyCategoricalEncoder() replaces categories with the number of observations or percentage
of observations per category. For example, if 10 observations show the category blue for the variable
color, blue will be replaced by 10. If, using frequency, if 20% of observations show the category red, 
red will be replaced by 0.20.

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

	# set up the encoder
	encoder = ce.CountFrequencyCategoricalEncoder(encoding_method='frequency',
				 variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	encoder.fit(X_train)

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)

	encoder.encoder_dict_


.. code:: python

	{'cabin': {'n': 0.7663755458515283,
	  'C': 0.07751091703056769,
	  'B': 0.04585152838427948,
	  'E': 0.034934497816593885,
	  'D': 0.034934497816593885,
	  'A': 0.018558951965065504,
	  'F': 0.016375545851528384,
	  'G': 0.004366812227074236,
	  'T': 0.001091703056768559},
	 'pclass': {3: 0.5436681222707423,
	  1: 0.25109170305676853,
	  2: 0.2052401746724891},
	 'embarked': {'S': 0.7117903930131004,
	  'C': 0.19759825327510916,
	  'Q': 0.0906113537117904}}


API Reference
-------------

.. autoclass:: feature_engine.categorical_encoders.CountFrequencyCategoricalEncoder
    :members: