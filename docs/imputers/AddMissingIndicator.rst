AddMissingIndicator
===================
The AddMissingIndicator() adds a binary variable indicating if observations are missing (missing indicator).
It adds a missing indicator for both categorical and numerical variables. A list of variables for which to
add a missing indicator can be passed, or the imputer will automatically select all variables.

The imputer has the option to select if binary variables should be added to all variables, or only to those
that show missing data in the train set, by setting the option how='missing_only'.

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
	addBinary_imputer = mdi.AddMissingIndicator(
	    variables=['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea'])

	# fit the imputer
	addBinary_imputer.fit(X_train)

	# transform the data
	train_t = addBinary_imputer.transform(X_train)
	test_t = addBinary_imputer.transform(X_test)

	train_t[['Alley_na', 'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']].head()

.. image:: ../images/missingindicator.png

API Reference
-------------

.. autoclass:: feature_engine.missing_data_imputers.AddMissingIndicator
    :members:
