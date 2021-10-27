.. _add_missing_indicator:

.. currentmodule:: feature_engine.imputation

AddMissingIndicator
===================


The :class:`AddMissingIndicator()` adds a binary variable indicating if observations are
missing (missing indicator). It adds missing indicators to both categorical and numerical
variables.

You can select the variables for which the missing indicators should be created passing
a variable list to the `variables` parameter. Alternatively, the imputer will
automatically select all variables.

The imputer has the option to add missing indicators to all variables or only to those
that have missing data in the train set. You can change the behaviour using the
parameter `missing_only`.

If `missing_only=True`, missing indicators will be added only to those variables with
missing data in the train set. This means that if you passed a variable list to
`variables` and some of those variables did not have missing data, no missing indicators
will be added to them. If it is paramount that all variables in your list get their
missing indicators, make sure to set `missing_only=False`.

It is recommended to use `missing_only=True` when not passing a list of variables to
impute.

Below a code example using the House Prices Dataset (more details about the dataset
:ref:`here <datasets>`).

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.imputation import AddMissingIndicator

	# Load dataset
	data = pd.read_csv('houseprice.csv')


	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
    	data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

	# set up the imputer
	addBinary_imputer = AddMissingIndicator(
        variables=['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea'],
        )

	# fit the imputer
	addBinary_imputer.fit(X_train)

	# transform the data
	train_t = addBinary_imputer.transform(X_train)
	test_t = addBinary_imputer.transform(X_test)

	train_t[['Alley_na', 'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']].head()

.. image:: ../../images/missingindicator.png

More details
^^^^^^^^^^^^

Check also this `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/imputation/AddMissingIndicator.ipynb>`_



