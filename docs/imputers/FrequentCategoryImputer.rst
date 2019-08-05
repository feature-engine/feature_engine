FrequentCategoryImputer
=======================
The FrequentCategoryImputer() replaces missing data in categorical variables with the most frequent
category. It works only with categorial variables. A list of variables can be indiacated, or the 
imputer will automatically select all categorical variables in the train set.

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
					data.drop(['Id', 'SalePrice'], axis=1),
					data['SalePrice'], test_size=0.3, random_state=0)

	# set up the imputer
	imputer = mdi.FrequentCategoryImputer(variables='MasVnrType')

	# fit the imputer
	imputer.fit(X_train)

	# transform the data
	train_t= imputer.transform(X_train)
	test_t= imputer.transform(X_test)


API Reference
-------------

.. autoclass:: feature_engine.missing_data_imputers.FrequentCategoryImputer
    :members: