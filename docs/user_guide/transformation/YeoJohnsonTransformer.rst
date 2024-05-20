.. _yeojohnson:

.. currentmodule:: feature_engine.transformation

YeoJohnsonTransformer
=====================

The Yeo-Johnson transformation which is an extension of the Box-Cox transformation is used on variables with zero and negative values as well as positive values. The caveat with the Box-Cox transformation is that it was designed only for numeric variables with positive values. If there are variables with negative values in the original data, we can either shift the variable distribution by adding a constant, or use the Yeo-Johnson transformation. 

If our variable X has strictly positive values, then, the Yeo-Johnson transformation is the same as the Box-Cox power transformation of (X + 1). If the variable X has strictly negative values, then the Yeo-Johnson transformation is same as the Box-Cox transformation of (-X + 1) but with power (2 — λ) where lambda is the transformation parameter. If the variable has both positive values and negative values, then the yeo-johnson power transformation is a mixture of two functions, and in that case different powers or parameters are used for the transforming the variable. 

For yeo-johnson power transformation of input features in a dataset, scipy.stats can be used with which only one variable is transformed at a time. But with other python libraries like sklearn and Feature-engine we can transform multiple variables simultaneously. There are differences between the sklearn and Feature-engine implementations of yeo-johnson power transformation.


The YeoJohnsonTransformer
--------------------------

The :class:`YeoJohnsonTransformer()` applies the Yeo-Johnson transformation to the numeric variables.

The Yeo-Johnson transformation is defined as:

.. image:: ../../images/yeojohnsonformula.png

where Y is the response variable and λ is the transformation parameter.

The Yeo-Johnson transformation implemented by this transformer is that of
`scipy.stats <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html>`_.

Python Implementation
----------------------

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

Now we apply the Yeo-Johnson transformation to the two (indicated) variables:

.. code:: python

	# set up the variable transformer
	tf = vt.YeoJohnsonTransformer(variables = ['LotArea', 'GrLivArea'])

	# fit the transformer
	tf.fit(X_train)

With `fit()`, the :class:`YeoJohnsonTransformer()` learns the optimal lambda for the yeo-johnson power transformation.
Now we can go ahead and transform the dataset to get closer to normal distributions. 

.. code:: python

	# transform the data
	train_t= tf.transform(X_train)
	test_t= tf.transform(X_test)

Next, we make a histogram of the original variable distribution:

.. code:: python

	# un-transformed variable
	X_train['LotArea'].hist(bins=50)

.. image:: ../../images/lotarearaw.png

Then, we explore the distribution of the transformed variable: 

.. code:: python

	# transformed variable
	train_t['LotArea'].hist(bins=50)


.. image:: ../../images/lotareayeojohnson.png

Note that the transformed variable has a more symmetric shape, gaussian-like distribution. 

Additional resources
--------------------

You can find more details about the :class:`YeoJohnsonTransformer()` here:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/transformation/YeoJohnsonTransformer.ipynb>`_

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

The differences between scipy.stats, sklearn and Feature-engine in implementing the yeo-johnson law of data transformation are highlighted in the book.

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.
