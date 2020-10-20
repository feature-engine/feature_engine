MeanMedianImputer
=================
The MeanMedianImputer() replaces missing data with the mean or median of the variable.
It works only with numerical variables. A list of variables to impute can be indicated,
or the imputer will automatically select all numerical variables in the train set. For
more details, check the API Reference below.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.imputation import MeanMedianImputer

	# Load dataset
	data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

	# set up the imputer
	median_imputer = MeanMedianImputer(imputation_method='median', variables=['LotFrontage', 'MasVnrArea'])

	# fit the imputer
	median_imputer.fit(X_train)

	# transform the data
	train_t= median_imputer.transform(X_train)
	test_t= median_imputer.transform(X_test)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	X_train['LotFrontage'].plot(kind='kde', ax=ax)
	train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
	lines, labels = ax.get_legend_handles_labels()
	ax.legend(lines, labels, loc='best')

.. image:: ../images/medianimputation.png

API Reference
-------------

.. autoclass:: feature_engine.imputation.MeanMedianImputer
    :members:
