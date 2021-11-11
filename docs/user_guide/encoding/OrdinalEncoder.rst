.. _ordinal_encoder:

.. currentmodule:: feature_engine.encoding

OrdinalEncoder
==============


The :class:`OrdinalEncoder()` replaces the categories by digits, starting from 0 to k-1,
where k is the number of different categories. If you select **"arbitrary"** in the
`encoding_method`, then the encoder will assign numbers as the labels appear in the
variable (first come first served). If you select **"ordered"**, the encoder will assign
numbers following the mean of the target value for that label. So labels for which the
mean of the target is higher will get the number 0, and those where the mean of the
target is smallest will get the number k-1. This way, we create a monotonic relationship
between the encoded variable and the target.

**Arbitrary vs ordered encoding**

**Ordered ordinal encoding**: for the variable colour, if the mean of the target
for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 1,
red by 2 and grey by 0.

The motivation is to try and create a monotonic relationship between the target and the
encoded categories. This tends to help improve performance of linear models.

**Arbitrary ordinal encoding**: the numbers will be assigned arbitrarily to the
categories, on a first seen first served basis.

Let's look at an example using the Titanic Dataset.

First, let's load the data and separate it into train and test:

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


Now, we set up the :class:`OrdinalEncoder()` to replace the categories by strings based
on the target mean value and only in the 3 indicated variables:

.. code:: python

	# set up the encoder
	encoder = OrdinalEncoder(encoding_method='ordered', variables=['pclass', 'cabin', 'embarked'])

	# fit the encoder
	encoder.fit(X_train, y_train)

With `fit()` the encoder learns the mappings for each category, which are stored in
its `encoder_dict_` parameter:

.. code:: python

	encoder.encoder_dict_

In the `encoder_dict_` we find the integers that will replace each one of the categories
of each variable that we want to encode. This way, we can map the original value of the
variable to the new value.

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

We can now go ahead and replace the original strings with the numbers:

.. code:: python

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)




More details
^^^^^^^^^^^^
In the following notebook, you can find more details into the :class:`OrdinalEncoder()`'s
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/OrdinalEncoder.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
