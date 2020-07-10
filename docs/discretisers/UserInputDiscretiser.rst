UserInputDiscretiser
====================
The UserInputDiscretiser() sorts the variable values into contiguous intervals which limits are arbitrarily
defined by the user.

The user must provide a dictionary of variable:list of limits pair when setting up the discretiser. 

The UserInputDiscretiser() works only with numerical variables. The discretiser will check that the variables
entered by the user are present in the train set and cast as numerical.

.. code:: python

	import numpy as np
	import pandas as pd
	from sklearn.datasets import load_boston
	from feature_engine.discretisers import UserInputDiscretiser

	boston_dataset = load_boston()
	data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

	user_dict = {'LSTAT': [0, 10, 20, 30, np.Inf]}

	transformer = UserInputDiscretiser(
	    binning_dict=user_dict, return_object=False, return_boundaries=False)
	X = transformer.fit_transform(data)

	X['LSTAT'].head()


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

	0    0
	1    0
	2    0
	3    0
	4    0
	Name: LSTAT, dtype: int64



API Reference
-------------

.. autoclass:: feature_engine.discretisers.EqualWidthDiscretiser
    :members: