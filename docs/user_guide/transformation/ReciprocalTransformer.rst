.. _reciprocal:

.. currentmodule:: feature_engine.transformation

ReciprocalTransformer
=====================

The :class:`ReciprocalTransformer()` applies the reciprocal transformation 1 / x to
numerical variables.

The :class:`ReciprocalTransformer()` only works with numerical variables with non-zero
values. If a variable contains the value 0, the transformer will raise an error.

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

Now we want to apply the reciprocal transformation to 2 variables in the dataframe:

.. code:: python

	# set up the variable transformer
	tf = vt.ReciprocalTransformer(variables = ['LotArea', 'GrLivArea'])

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

Additional resources
--------------------

You can find more details about the :class:`ReciprocalTransformer()` here:


- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/transformation/ReciprocalTransformer.ipynb>`_

For more details about this and other feature engineering methods check out these resources:


.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

|
|
|
|
|
|
|
|
|
|

Or read our book:

.. figure::  ../../images/cookbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://packt.link/0ewSo

   Python Feature Engineering Cookbook

|
|
|
|
|
|
|
|
|
|
|
|
|

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.