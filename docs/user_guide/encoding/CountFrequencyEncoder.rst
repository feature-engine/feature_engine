.. _count_freq_encoder:

.. currentmodule:: feature_engine.encoding

CountFrequencyEncoder
=====================

The :class:`CountFrequencyEncoder()` replaces categories by either the count or the
percentage of observations per category. For example in the variable colour, if 10
observations are blue, blue will be replaced by 10. Alternatively, if 10% of the
observations are blue, blue will be replaced by 0.1.

Let's look at an example using the Titanic Dataset.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import CountFrequencyEncoder

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
	encoder = CountFrequencyEncoder(encoding_method='frequency',
				 variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	encoder.fit(X_train)

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)

	encoder.encoder_dict_

In the `encoder_dict_` we find the frequencies for each one of the categories of each
variable that we want to encode. This way, we can map the original value to the new
value.

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


More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/CountFrequencyEncoder.ipynb>`_
