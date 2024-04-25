.. _equal_width_discretiser:

.. currentmodule:: feature_engine.discretisation

EqualWidthDiscretiser
=====================

The :class:EqualWidthDiscretiser() divides continuous numerical variables into intervals of equal width, calculated using the formula:

( max(X) - min(X) ) / bins

where bins is the number of intervals specified by the user. This discretisation is achieved through the pandas.cut() function.

Discretization is a common data preprocessing technique used in data science. It's also known as data binning (or simply `binning`).

Advantages and Limitations
--------------------------

Advantages
~~~~~~~~~~

Some advantagers of equal width binning:

- **Algorithm Efficiency:** Enhances the performance of data mining and machine learning algorithms by providing a simplified representation of the dataset.
- **Outlier Management:** Efficiently mitigates the effect of outliers by grouping them into the extreme bins, thus preserving the integrity of the main data distribution.
- **Data Smoothing:** Helps smooth the data, reduces noise, and improves the model's ability to generalize.

Limitations
~~~~~~~~~~~

On the other hand, :class:`EqualWidthDiscretiser()` can lead to a loss of information by aggregating data into broader categories. This is particularly concerning if the data in the same bin has predictive information about the target.

Let's consider a binary classifier task using a decision tree model. A bin with a high proportion of both categories would potentially impact the model's performance in this scenario.

Notes
-----

`EqualWidthDiscretiser` expects a `pandas.DataFrame` and works only with numerical variables. The user can specify the variables to be discretized. Otherwise, `EqualWidthDiscretiser` will automatically infer the data types to compute the interval limits for all numeric variables.

**Optimal number of intervals:** With `EqualWidthDiscretiser`, the user defines the number of bins. Smaller intervals may be required if the variable is highly skewed or not continuous. Otherwise, the transformer will introduce `numpy.nan`.

**Integration with scikit-learn:** `EqualFrequencyDiscretiser` and all other feature-engine transformers seamlessly integrate with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [column transformers](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).

Python code example
-------------------

Load dataset
~~~~~~~~~~~~

In this example, we'll use the House Prices' Dataset (for more details, please check [this link](https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=house_prices)).

First, let's load the dataset and split it into train and test sets:

.. code:: python

	import matplotlib.pyplot as plt
	from sklearn.datasets import fetch_openml
	from sklearn.model_selection import train_test_split

	from feature_engine.discretisation import EqualFrequencyDiscretiser

	# Load dataset
	X, y = fetch_openml(name='house_prices', version=1, return_X_y=True, as_frame=True)
	X.set_index('Id', inplace=True)

	# Separate into train and test sets
	X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=42)


Equal-width Discretisation
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, let's discretize two variables (LotArea and GrLivArea) into 10 intervals of equal width:

.. code:: python

	# List the target numeric variables for equal-width discretization
	TARGET_NUMERIC_FEATURES= ['LotArea','GrLivArea']

	# Set up the discretisation transformer
	disc = EqualWidthDiscretiser(bins=10, variables=TARGET_NUMERIC_FEATURES)

	# Fit the transformer
	disc.fit(X_train)


Note that if we do not specify the variables (default=`None`), `EqualWidthDiscretiser` will automatically infer the data types to compute the interval limits for all numeric variables.

With the `.fit()` method, the discretiser learns the bin boundaries and saves them into a dictionary so we can use them to transform unseen data:

.. code:: python

	# Learnt limits for each variable
	disc.binner_dict_


.. code:: python

	{'LotArea': [-inf,
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
	  864.8,
	  1395.6,
	  1926.3999999999999,
	  2457.2,
	  2988.0,
	  3518.7999999999997,
	  4049.5999999999995,
	  4580.4,
	  5111.2,
	  inf]}


Note that the lower and upper boundariers are set to -inf and inf, respectively. This behavior ensures the transformer works even for unseen limits (lower than the minimum or greater than the maximum trained value).

Also, this transformer will not work in the presence of missing values. Therefore, we should either remove or impute missing values before fitting the transformer.

.. code:: python

	# Transform the data (data discretization)
	train_t = disc.transform(X_train)
	test_t = disc.transform(X_test)

Let's visualize the first rows of the raw data and the transformed data:

.. code:: python

	# Raw data
	print(X_train[TARGET_NUMERIC_FEATURES].head())


.. code:: python

		  LotArea  GrLivArea
	Id                      
	136     10400       1682
	1453     3675       1072
	763      8640       1547
	933     11670       1905
	436     10667       1661


.. code:: python

	# Transformed data
	print(train_t[TARGET_NUMERIC_FEATURES].head())


.. code:: python

		  LotArea  GrLivArea
	Id                      
	136         0          2
	1453        0          1
	763         0          2
	933         0          2
	436         0          2


The transformed data now contains discrete values corresponding to the ordered computed buckets (0 being the first and bins-1 the last).

Now, let's visualize its output:

.. code:: python

	train_t['GrLivArea'].value_counts().sort_index().plot.bar()
	plt.ylabel('Number of houses')
	plt.show()


.. image:: ../../images/equalwidthdiscretisation.png

As we can see, the intervals contain different number of observations.  
It's a similar output of an histogram, but it's designed to work transform unseen data, includig outliers.

Finally, since the default value for the `return_object` parameter is `False`, the transformer outputs integer variables:

.. code:: python

	train_t[TARGET_NUMERIC_FEATURES].dtypes


.. code:: python

	LotArea      int64
	GrLivArea    int64
	dtype: object


Return object instead of integers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Categorical encoders in feature-engine are designed to work by default with variables of type object. Therefore, to further encode the discretiser output with feature-engine, we can set `return_object=True` instead. This will return the transformed variables as object.

Let's say we want to obtain monotonic relationships between the variable and the target. We can do that seamlessly by setting `return_object` to True. A tutorial of how to use this functionality is available [here](https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/discretisation/EqualWidthDiscretiser_plus_OrdinalEncoder.ipynb).

Additionally, if we want to output the intervals as object while specifying the boundaries, we can set `return_boundaries` to True:

.. code:: python

	# Set up the discretisation transformer
	disc = EqualFrequencyDiscretiser(bins=10, variables=TARGET_NUMERIC_FEATURES, return_boundaries=True)

	# Fit the transformer
	disc.fit(X_train)

	# Transform test set & visualize limit
	test_t = disc.transform(X_test)

	# Visualize output (boundaries)
	print(test_t[TARGET_NUMERIC_FEATURES].head())


.. code:: python

		     LotArea         GrLivArea
	Id                                     
	893   (-inf, 22694.5]   (864.8, 1395.6]
	1106  (-inf, 22694.5]  (2457.2, 2988.0]
	414   (-inf, 22694.5]   (864.8, 1395.6]
	523   (-inf, 22694.5]  (1395.6, 1926.4]
	1037  (-inf, 22694.5]  (1395.6, 1926.4]


See Also
--------

- Further feature-engine discretiser / binning methods[here](https://feature-engine.trainindata.com/en/latest/user_guide/discretisation/index.html)
- Scikit-learn [`KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer) class
- [Pandas cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html)

Additional resources
--------------------

Check also for more details on how to use this transformer:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/discretisation/EqualWidthDiscretiser.ipynb>`_
- `Jupyter notebook - Discretiser plus Ordinal encoding <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/discretisation/EqualWidthDiscretiser_plus_OrdinalEncoder.ipynb>`_

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