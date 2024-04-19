.. _mean_median_imputer:

.. currentmodule:: feature_engine.imputation

MeanMedianImputer
=================

The :class:`MeanMedianImputer()` replaces missing data with the mean or median of the variable.
It works only with numerical variables. You can pass the list of variables you want to impute,
or alternatively, the imputer will automatically select all numerical variables in the
train set.

Note that in symetrical distributions, the mean and the median are very similar. But in
skewed distributions, the median is a better representation of the majority, as the mean
is biased to extreme values. The following image was taken from Wikipedia. The image links
to the use license.

.. figure::  ../../images/1024px-Relationship_between_mean_and_median_under_different_skewness.png
   :align:   center
   :target: https://commons.wikimedia.org/wiki/File:Relationship_between_mean_and_median_under_different_skewness.png


With the `fit()` method, the transformer learns and stores the mean or median values per
variable. Then it uses these values in the `transform()` method to transform the data.

Below a code example using the House Prices Dataset (more details about the dataset
:ref:`here <datasets>`).

First, let's load the data and separate it into train and test:

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
                                            data.drop(['Id', 'SalePrice'], axis=1),
                                            data['SalePrice'],
                                            test_size=0.3,
                                            random_state=0,
                                            )


Now we set up the :class:`MeanMedianImputer()` to impute in this case with the median
and only 2 variables from the dataset.

.. code:: python

	# set up the imputer
	median_imputer = MeanMedianImputer(
                           imputation_method='median',
                           variables=['LotFrontage', 'MasVnrArea']
                           )

	# fit the imputer
	median_imputer.fit(X_train)

With fit, the :class:`MeanMedianImputer()` learned the median values for the indicated
variables and stored it in one of its attributes. We can now go ahead and impute both
the train and the test sets.

.. code:: python

	# transform the data
	train_t= median_imputer.transform(X_train)
	test_t= median_imputer.transform(X_test)

Note that after the imputation, if the percentage of missing values is relatively big,
the variable distribution will differ from the original one (in red the imputed
variable):

.. code:: python

	fig = plt.figure()
	ax = fig.add_subplot(111)
	X_train['LotFrontage'].plot(kind='kde', ax=ax)
	train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
	lines, labels = ax.get_legend_handles_labels()
	ax.legend(lines, labels, loc='best')

.. image:: ../../images/medianimputation.png

Additional resources
--------------------

In the following Jupyter notebook you will find more details on the functionality of the
:class:`MeanMedianImputer()`, including how to select numerical variables automatically.
You will also see how to navigate the different attributes of the transformer to find the
mean or median values of the variables.

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/imputation/MeanMedianImputer.ipynb>`_

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