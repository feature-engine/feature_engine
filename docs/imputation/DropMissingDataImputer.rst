DropMissingDataImputer
=====================

API Reference
-------------

.. autoclass:: feature_engine.imputation.DropMissingData
    :members:

Example
-------

The DropMissingData() deletes rows with NA values. It works  numerical and categorical 
variables. A list of variables for which to delete rows with NA values can be passed, or
the imputer will automatically select all variables

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.imputation import DropMissingData

	# Load dataset
	data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

	# set up the imputer
	missingdata_imputer = DropMissingData(variables=['LotFrontage', 'MasVnrArea'])

	# fit the imputer
	missingdata_imputer.fit(X_train)

	# transform the data
	train_t= missingdata_imputer.transform(X_train)
	test_t= missingdata_imputer.transform(X_test)

    # No of NA's before and after transformation
	X_train['LotFrontage'].isna().sum()
	train_t['LotFrontage'].isna().sum()




