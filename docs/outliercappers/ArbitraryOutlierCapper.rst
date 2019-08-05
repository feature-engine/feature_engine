ArbitraryOutlierCapper
======================

API Reference
-------------

.. autoclass:: feature_engine.outlier_removers.ArbitraryOutlierCapper
    :members:


Example Use
-----------

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine import outlier_removers as outr

	# Load dataset
	def load_titanic():
		data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
		data = data.replace('?', np.nan)
		data['cabin'] = data['cabin'].astype(str).str[0]
		data['pclass'] = data['pclass'].astype('O')
		data['embarked'].fillna('C', inplace=True)
		data['fare'] = data['fare'].astype('float')
		data['age'] = data['age'].astype('float')
		return data
	
	data = load_titanic()

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
			data.drop(['survived', 'name', 'ticket'], axis=1),
			data['survived'], test_size=0.3, random_state=0)

	# set up the capper
	capper = outr.ArbitraryOutlierCapper(
			max_capping_dict={'age': 50, 'fare': 200}, min_capping_dict=None)

	# fit the capper
	capper.fit(X_train)

	# transform the data
	train_t= capper.transform(X_train)
	test_t= capper.transform(X_test)

	capper.right_tail_caps_


.. code:: python

	{'age': 50, 'fare': 200}

