EqualWidthDiscretiser
=====================
The EqualWidthDiscretiser() sorts the variable values into contiguous intervals of equal size. The size
of the interval is calculated as:

( max(X) - min(X) ) / bins

where bins, which is the number of intervals, should be determined by the user. The transformer can return
the variable as numeric or object (default = numeric).

The EqualWidthDiscretiser() works only with numerical variables. A list of variables can
be indicated, or the imputer will automatically select all numerical variables in the train set.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine import discretisers as dsc

	# Load dataset
	data = data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test =  train_test_split(
		    data.drop(['Id', 'SalePrice'], axis=1),
		    data['SalePrice'], test_size=0.3, random_state=0)

	# set up the discretisation transformer
	disc = dsc.EqualWidthDiscretiser(bins=10, variables=['LotArea', 'GrLivArea'])

	# fit the transformer
	disc.fit(X_train)

	# transform the data
	train_t= disc.transform(X_train)
	test_t= disc.transform(X_test)

	disc.binner_dict_


.. code:: python

	'LotArea': [-inf,
	  22694.5,
	  44089.0,
	  65483.5,
	  86878.0,
	  108272.5,
	  129667.0,
	  151061.5,
	  172456.0,
	  193850.5,
	  inf],
	 'GrLivArea': [-inf,
	  768.2,
	  1202.4,
	  1636.6,
	  2070.8,
	  2505.0,
	  2939.2,
	  3373.4,
	  3807.6,
	  4241.799999999999,
	  inf]}


.. code:: python

	# with equal width discretisation, each bin does not necessarily contain
	# the same number of observations.
	train_t.groupby('GrLivArea')['GrLivArea'].count().plot.bar()
	plt.ylabel('Number of houses')


.. image:: ../images/equalwidthdiscretisation.png


API Reference
-------------

.. autoclass:: feature_engine.discretisers.EqualWidthDiscretiser
    :members: