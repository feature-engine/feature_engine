EqualFrequencyDiscretiser
=========================
The EqualFrequencyDiscretiser() sorts the variable values into contiguous intervals of equal proportion
of observations. The limits of the intervals are calculated according to the quantiles. The number of
intervals or quantiles should be determined by the user. The transformer can return the variable as
numeric or object (default = numeric).

The EqualFrequencyDiscretiser() works only with numerical variables. A list of variables can
be indiacated, or the imputer will automatically select all numerical variables in the train set.

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
	disc = dsc.EqualFrequencyDiscretiser(q=10, variables=['LotArea', 'GrLivArea'])

	# fit the transformer
	disc.fit(X_train)

	# transform the data
	train_t= disc.transform(X_train)
	test_t= disc.transform(X_test)

	disc.binner_dict_


.. code:: python

	{'LotArea': [-inf,
	  5007.1,
	  7164.6,
	  8165.700000000001,
	  8882.0,
	  9536.0,
	  10200.0,
	  11046.300000000001,
	  12166.400000000001,
	  14373.9,
	  inf],
	 'GrLivArea': [-inf,
	  912.0,
	  1069.6000000000001,
	  1211.3000000000002,
	  1344.0,
	  1479.0,
	  1603.2000000000003,
	  1716.0,
	  1893.0000000000005,
	  2166.3999999999996,
	  inf]}


.. code:: python

	# with equal frequency discretisation, each bin contains approximately
	# the same number of observations.
	train_t.groupby('GrLivArea')['GrLivArea'].count().plot.bar()
	plt.ylabel('Number of houses')


.. image:: ../images/equalfrequencydiscretisation.png


API Reference
-------------

.. autoclass:: feature_engine.discretisation.EqualFrequencyDiscretiser
    :members: