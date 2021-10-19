EndTailImputer
==============


The EndTailImputer() replaces missing data with a value at the end of the distribution.
The value can be determined using the mean plus or minus a number of times the standard
deviation, or using the inter-quartile range proximity rule. The value can also be
determined as a factor of the maximum value. See the API Reference above for more
details.

You decide whether the missing data should be placed at the right or left tail of
the variable distribution.

It works only with numerical variables. A list of variables can be indicated, or the
imputer will automatically select all numerical variables in the train set.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.imputation import EndTailImputer

	# Load dataset
	data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

	# set up the imputer
	tail_imputer = EndTailImputer(imputation_method='gaussian',
                                  tail='right',
                                  fold=3,
                                  variables=['LotFrontage', 'MasVnrArea'])
	# fit the imputer
	tail_imputer.fit(X_train)

	# transform the data
	train_t= tail_imputer.transform(X_train)
	test_t= tail_imputer.transform(X_test)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	X_train['LotFrontage'].plot(kind='kde', ax=ax)
	train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
	lines, labels = ax.get_legend_handles_labels()
	ax.legend(lines, labels, loc='best')

.. image:: ../../images/endtailimputer.png


