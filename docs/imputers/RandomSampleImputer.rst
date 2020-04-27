RandomSampleImputer
===================
The RandomSampleImputer() replaces missing data with a random sample extracted from the variable.
It works with both numerical and categorical variables. A list of variables can be indicated, or
the imputer will automatically select all variables in the train set.

A seed can be set to a pre-defined number and all observations will be replaced in batch. Alternatively,
a seed can be set using the values of 1 or more numerical variables. In this case, the observations will be
imputed individually, one at a time, using the values of the variables as a seed.

For example, if the observation shows variables color: np.nan, height: 152, weight:52, and we set
the imputer as:

.. code:: python

	RandomSampleImputer(random_state=['height', 'weight'],
                                  seed='observation',
                                  seeding_method='add'))

the observation will be replaced using pandas sample as follows:

.. code:: python

	observation.sample(1, random_state=int(152+52))

More details on how to use the RandomSampleImputer():

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	import feature_engine.missing_data_imputers as mdi

	# Load dataset
	data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

	# set up the imputer
	imputer = mdi.RandomSampleImputer(random_state=['MSSubClass', 'YrSold'],
                                  seed='observation',
                                  seeding_method='add')
	# fit the imputer
	imputer.fit(X_train)

	# transform the data
	train_t = imputer.transform(X_train)
	test_t = imputer.transform(X_test)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	X_train['LotFrontage'].plot(kind='kde', ax=ax)
	train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
	lines, labels = ax.get_legend_handles_labels()
	ax.legend(lines, labels, loc='best')

.. image:: ../images/randomsampleimputation.png

API Reference
-------------

.. autoclass:: feature_engine.missing_data_imputers.RandomSampleImputer
    :members:
