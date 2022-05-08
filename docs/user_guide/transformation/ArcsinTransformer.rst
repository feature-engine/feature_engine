.. _arcsin:

.. currentmodule:: feature_engine.transformation

ArcsinTransformer
=================

The :class:`ArcsinTransformer()` applies the arcsin transformation to
numerical variables.

The :class:`ArcsinTransformer()` only works with numerical variables with values between -1 and +1. If the variable contains a value outside of this range, the transformer will raise an error.

Let's load the house prices dataset and  separate it into train and test sets (more
details about the dataset :ref:`here <datasets>`).

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine import transformation as vt

	# Load dataset
	data = data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test =  train_test_split(
		    data.drop(['Id', 'SalePrice'], axis=1),
		    data['SalePrice'], test_size=0.3, random_state=0)

Now we want to apply the arcsin transformation to 2 variables in the dataframe:

.. code:: python

	# set up the variable transformer
	tf = vt.ArcsinTransformer(variables = ['LotArea', 'GrLivArea'])

	# fit the transformer
	tf.fit(X_train)

The transformer does not learn any parameters. So we can go ahead and transform the
variables:

.. code:: python

	# transform the data
	train_t= tf.transform(X_train)
	test_t= tf.transform(X_test)

Finally, we can plot the original variable distribution:

.. code:: python

	# un-transformed variable
	X_train['LotArea'].hist(bins=50)

.. image:: ../../images/lotarearaw.png

And now the distribution after the transformation:

.. code:: python

	# transformed variable
	train_t['LotArea'].hist(bins=50)


.. image:: ../../images/lotareareciprocal.png

More details
^^^^^^^^^^^^

You can find more details about the :class:`ArcsinTransformer()` here:


- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/transformation/ReciprocalTransformer.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
