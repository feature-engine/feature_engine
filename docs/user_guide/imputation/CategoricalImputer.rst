.. _categorical_imputer:

.. currentmodule:: feature_engine.imputation

CategoricalImputer
==================

The :class:`CategoricalImputer()` replaces missing data in categorical variables with an
arbitrary value, like the string 'Missing' or by the most frequent category.

Below a code example using the House Prices Dataset (more details about the dataset
:ref:`here <datasets>`).

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.imputation import CategoricalImputer

	# Load dataset
	data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

	# set up the imputer
	imputer = CategoricalImputer(variables=['Alley', 'MasVnrType'])

	# fit the imputer
	imputer.fit(X_train)

	# transform the data
	train_t= imputer.transform(X_train)
	test_t= imputer.transform(X_test)

	test_t['MasVnrType'].value_counts().plot.bar()

.. image:: ../../images/missingcategoryimputer.png

More details
^^^^^^^^^^^^

Check also this `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/imputation/CategoricalImputer.ipynb>`_

