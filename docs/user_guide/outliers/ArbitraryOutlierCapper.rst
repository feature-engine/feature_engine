.. _arbitrary_capper:

.. currentmodule:: feature_engine.outliers

ArbitraryOutlierCapper
======================

The :class:`ArbitraryOutlierCapper()` caps the maximum or minimum values of a variable
at an arbitrary value indicated by the user. The maximum or minimum values should be
entered in a dictionary with the form {feature:capping value}.

Let's look at this in an example. First we load the Titanic dataset, and separate it
into a train and a test set:

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.outliers import ArbitraryOutlierCapper

	# Load dataset
	def load_titanic():
		data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
		data = data.replace('?', np.nan)
		data['cabin'] = data['cabin'].astype(str).str[0]
		data['pclass'] = data['pclass'].astype('O')
		data['embarked'].fillna('C', inplace=True)
		data['fare'] = data['fare'].astype('float')
		data['fare'].fillna(data['fare'].median(), inplace=True)
		data['age'] = data['age'].astype('float')
		data['age'].fillna(data['age'].median(), inplace=True)
		return data
	
	data = load_titanic()

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
			data.drop(['survived', 'name', 'ticket'], axis=1),
			data['survived'], test_size=0.3, random_state=0)

Now, we set up the :class:`ArbitraryOutlierCapper()` indicating that we want to cap the
variable 'age' at 50 and the variable 'Fare' at 200. We do not want to cap these variables
on the left side of their distribution.

.. code:: python

	# set up the capper
	capper = ArbitraryOutlierCapper(max_capping_dict={'age': 50, 'fare': 200}, min_capping_dict=None)

	# fit the capper
	capper.fit(X_train)

With `fit()` the transformer does not learn any parameter. It just reassigns the entered
dictionary to the attribute that will be used in the transformation:

.. code:: python

	capper.right_tail_caps_

.. code:: python

	{'age': 50, 'fare': 200}

Now, we can go ahead and cap the variables:

.. code:: python

	# transform the data
	train_t= capper.transform(X_train)
	test_t= capper.transform(X_test)

If we now check the maximum values in the transformed data, they should be those entered
in the dictionary:

.. code:: python

    train_t[['fare', 'age']].max()

.. code:: python

    fare    200
    age      50
    dtype: float64


More details
^^^^^^^^^^^^

You can find more details about the :class:`ArbitraryOutlierCapper()` functionality in the following
notebook:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/outliers/ArbitraryOutlierCapper.ipynb>`_

