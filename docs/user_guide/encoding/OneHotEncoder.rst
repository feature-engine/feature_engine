.. _onehot_encoder:

.. currentmodule:: feature_engine.encoding

OneHotEncoder
=============

The :class:`OneHotEncoder()` performs one hot encoding. One hot encoding consists in
replacing the categorical variable by a group of binary variables which take value 0 or
1, to indicate if a certain category is present in an observation. The binary variables
are also known as dummy variables.

For example, from the categorical variable "Gender" with categories "female" and
"male", we can generate the boolean variable "female", which takes 1 if the
observation is female or 0 otherwise. We can also generate the variable "male",
which takes 1 if the observation is "male" and 0 otherwise. By default, the
:class:`OneHotEncoder()` will return both binary variables from "Gender": "female" and
"male".

**Binary variables**

When a categorical variable has only 2 categories, like "Gender" in our example, then
the second dummy variable created by one hot encoding can be completely redundant. We
can drop automatically the last dummy variable for those variables that contain only 2
categories by setting the parameter 'drop_last_binary=True`. This will ensure that for
every binary variable in the dataset, only 1 dummy is created. This is recommended,
unless we suspect that the variable could, in principle take more than 2 values.

**k vs k-1 dummies**

From a categorical variable with k unique categories, the :class:`OneHotEncoder()` can
create k binary variables, or alternatively k-1 to avoid redundant information. This
behaviour can be specified using the parameter `drop_last`. Only k-1 binary variables
are necessary to encode all of the information in the original variable. However, there
are situations in which we may choose to encode the data into k dummies.

Encode into k-1 if training linear models: Linear models evaluate all features during
fit, thus, with k-1 the have all information about the original categorical variable.

Encode into k if training decision trees or performing feature selection: tree based
models and many feature selection algorithms evaluate variables or groups of variables
separately. Thus, if encoding into k-1, the last category will not be examined. That is,
we lose the information contained in that category.

**Encoding only popular categories**

The encoder can also create binary variables for the n most popular categories, n being
determined by the user. This means, if we encode the 6 more popular categories, we will
only create binary variables for those categories, and the rest will be dropped. This is
useful when the categorical variables are highly cardinal, to control the expansion of
the feature space.

**Note**

Only when creating binary variables for all categories of the variable (instead of the
most popular ones), we can specify if we want to encode into k or k-1 binary variables,
where k is the number if unique categories. If we encode only the top n most popular
categories, the encoder will create only n binary variables per categorical variable.
Observations that do not show any of these popular categories, will have 0 in all
the binary variables.

Let's look at an example using the Titanic Dataset.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import OneHotEncoder

	# Load dataset
	def load_titanic():
		data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
		data = data.replace('?', np.nan)
		data['cabin'] = data['cabin'].astype(str).str[0]
		data['pclass'] = data['pclass'].astype('O')
		data['embarked'].fillna('C', inplace=True)
		return data
	
	data = load_titanic()

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
				data.drop(['survived', 'name', 'ticket'], axis=1),
				data['survived'], test_size=0.3, random_state=0)

	# set up the encoder
	encoder = OneHotEncoder( top_categories=2, variables=['pclass', 'cabin', 'embarked'], drop_last=False)

	# fit the encoder
	encoder.fit(X_train)

	# transform the data
	train_t= encoder.transform(X_train)
	test_t= encoder.transform(X_test)

	encoder.encoder_dict_

The `encoder_dict_` contains the categories that will derive dummy variables for each
categorical variable.

.. code:: python

	{'pclass': [3, 1], 'cabin': ['n', 'C'], 'embarked': ['S', 'C']}

**Feature space and duplication**

If the categorical variables are highly cardinal, we may end up with very big datasets.
In addition, if some of these variables are fairly constant or fairly similar, we may
end up with one hot encoded features that are highly correlated if not identical.

Consider checking this up and dropping redundant features with the transformers from the
:ref:`selection module <selection_user_guide>`.

More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/OneHotEncoder.ipynb>`_
