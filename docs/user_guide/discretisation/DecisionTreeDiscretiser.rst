DecisionTreeDiscretiser
=======================

In the original article, each feature of the challenge dataset was recoded by training
a decision tree of limited depth (2, 3 or 4) using that feature alone, and letting the
tree predict the target. The probabilistic predictions of this decision tree were used
as an additional feature, that was now linearly (or at least monotonically) correlated
with the target.

According to the authors, the addition of these new features had a significant impact
on the performance of linear models.

In the following example, we recode 2 numerical variables using decision trees.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.discretisation import DecisionTreeDiscretiser

	# Load dataset
	data = data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test =  train_test_split(
		    data.drop(['Id', 'SalePrice'], axis=1),
		    data['SalePrice'], test_size=0.3, random_state=0)

	# set up the discretisation transformer
	disc = DecisionTreeDiscretiser(cv=3,
                                  scoring='neg_mean_squared_error',
                                  variables=['LotArea', 'GrLivArea'],
                                  regression=True)

	# fit the transformer
	disc.fit(X_train, y_train)

	# transform the data
	train_t= disc.transform(X_train)
	test_t= disc.transform(X_test)

	disc.binner_dict_


.. code:: python

	{'LotArea': GridSearchCV(cv=3, error_score='raise-deprecating',
	              estimator=DecisionTreeRegressor(criterion='mse', max_depth=None,
	                                              max_features=None,
	                                              max_leaf_nodes=None,
	                                              min_impurity_decrease=0.0,
	                                              min_impurity_split=None,
	                                              min_samples_leaf=1,
	                                              min_samples_split=2,
	                                              min_weight_fraction_leaf=0.0,
	                                              presort=False, random_state=None,
	                                              splitter='best'),
	              iid='warn', n_jobs=None, param_grid={'max_depth': [1, 2, 3, 4]},
	              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
	              scoring='neg_mean_squared_error', verbose=0),
	 'GrLivArea': GridSearchCV(cv=3, error_score='raise-deprecating',
	              estimator=DecisionTreeRegressor(criterion='mse', max_depth=None,
	                                              max_features=None,
	                                              max_leaf_nodes=None,
	                                              min_impurity_decrease=0.0,
	                                              min_impurity_split=None,
	                                              min_samples_leaf=1,
	                                              min_samples_split=2,
	                                              min_weight_fraction_leaf=0.0,
	                                              presort=False, random_state=None,
	                                              splitter='best'),
	              iid='warn', n_jobs=None, param_grid={'max_depth': [1, 2, 3, 4]},
	              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
	              scoring='neg_mean_squared_error', verbose=0)}


.. code:: python

	# with tree discretisation, each bin does not necessarily contain
	# the same number of observations.
	train_t.groupby('GrLivArea')['GrLivArea'].count().plot.bar()
	plt.ylabel('Number of houses')


.. image:: ../../images/treediscretisation.png


