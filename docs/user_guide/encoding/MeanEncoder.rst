.. _mean_encoder:

.. currentmodule:: feature_engine.encoding

MeanEncoder
===========

The :class:`MeanEncoder()` replaces categories with the mean of the target per category.
For example, if we are trying to predict default rate, and our data has the variable city,
with categories, London, Manchester and Bristol, and the default rate per city is 0.1,
0.5, and 0.3, respectively, the encoder will replace London by 0.1, Manchester by 0.5
and Bristol by 0.3.

The motivation is to try and create a monotonic relationship between the target and
the encoded categories. This tends to help improve performance of linear models.

On top of that :class:`MeanEncoder()` provides option for smoothing (parameter a), which
allow user to even out (smooth) output of encoder for categories with different number 
of observations. 
It works by assigning weighted mean of the whole target column (prior) and mean of target within 
each category (evidence), instead of just evidence. The more samples of the specific category 
is avaliable, the bigger weight will be assigned to evidence, less samples - bigger weight to prior.
This parameter could be used as form of regularization. By default a is equal to 0, so no 
smoothing is applied.
Another option would be to use 'auto' heuristic, which will determine smoothing factor 
based on ratio between variance of target within each category to variance of the whole target column.

Let's look at an example using the Titanic Dataset.

First, let's load the data and separate it into train and test:

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

Now, we set up the :class:`MeanEncoder()` to replace the categories only in the 3
indicated variables:

.. code:: python

	# set up the encoder
	encoder = MeanEncoder(variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	encoder.fit(X_train, y_train)

With `fit()` the encoder learns the target mean value for each category, which are stored in
its `encoder_dict_` parameter:

.. code:: python

	encoder.encoder_dict_

The `encoder_dict_` contains the mean value of the target per category, per variable.
So we can easily use this dictionary to map the numbers to the original labels.

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

We can now go ahead and replace the original strings with the numbers:

.. code:: python

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)


Handling Cardinality
^^^^^^^^^^^^^^^^^^^^

The :class:`MeanEncoder()` replaces categories with the mean of the target per category.
If the variable has low cardinality, then there is a fair representation of each label
in the dataset, and the mean target value per category can be determined with some certainty.
However, if variables are highly cardinal, with only very few observations for some labels,
then the mean target value for those categories will be unreliable.

To encode highly cardinal variables using target mean encoding, we could either group
infrequent categories first using the :class:`RareLabelEncoder()`. Alternatively, we
may want to choose different encoding methods that use blends of probabilities to try and
better estimate the encoding mappings, like those available in the open-source package
Category encoders through the transformers
`M-estimate <https://contrib.scikit-learn.org/category_encoders/mestimate.html>`_ and
`Target Encoder <https://contrib.scikit-learn.org/category_encoders/targetencoder.html>`_.


More details
^^^^^^^^^^^^

In the following notebook, you can find more details into the :class:`MeanEncoder()`
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/MeanEncoder.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
