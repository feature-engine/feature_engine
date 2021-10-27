.. _woe_encoder:

.. currentmodule:: feature_engine.encoding

WoEEncoder
==========

The :class:`WoEEncoder()` replaces categories by the weight of evidence
(WoE). The WoE was used primarily in the financial sector to create credit risk
scorecards.

The weight of evidence is given by:

.. math::

    log( p(X=xj|Y = 1) / p(X=xj|Y=0) )



**The WoE is determined as follows:**

We calculate the percentage positive cases in each category of the total of all
positive cases. For example 20 positive cases in category A out of 100 total
positive cases equals 20 %. Next, we calculate the percentage of negative cases in
each category respect to the total negative cases, for example 5 negative cases in
category A out of a total of 50 negative cases equals 10%. Then we calculate the
WoE by dividing the category percentages of positive cases by the category
percentage of negative cases, and take the logarithm, so for category A in our
example WoE = log(20/10).

**Note**

- If WoE values are negative, negative cases supersede the positive cases.
- If WoE values are positive, positive cases supersede the negative cases.
- And if WoE is 0, then there are equal number of positive and negative examples.

**Encoding into WoE**:

- Creates a monotonic relationship between the encoded variable and the target
- Returns variables in a similar scale

Let's look at an example using the Titanic Dataset.

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.encoding import WoEEncoder, RareLabelEncoder

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

	# set up a rare label encoder
	rare_encoder = RareLabelEncoder(tol=0.03, n_categories=2, variables=['cabin', 'pclass', 'embarked'])

	# fit and transform data
	train_t = rare_encoder.fit_transform(X_train)
	test_t = rare_encoder.transform(X_train)

	# set up a weight of evidence encoder
	woe_encoder = WoEEncoder(variables=['cabin', 'pclass', 'embarked'])

	# fit the encoder
	woe_encoder.fit(train_t, y_train)

	# transform
	train_t = woe_encoder.transform(train_t)
	test_t = woe_encoder.transform(test_t)

	woe_encoder.encoder_dict_

In the `encoder_dict_` we find the WoE for each one of the categories of the
variables to encode. This way, we can map the original values to the new value.

.. code:: python

    {'cabin': {'B': 1.6299623810120747,
    'C': 0.7217038208351837,
    'D': 1.405081209799324,
    'E': 1.405081209799324,
    'Rare': 0.7387452866900354,
    'n': -0.35752781962490193},
    'pclass': {1: 0.9453018143294478,
    2: 0.21009172435857942,
    3: -0.5841726684724614},
    'embarked': {'C': 0.6999054533737715,
    'Q': -0.05044494288988759,
    'S': -0.20113381737960143}}


**WoE for continuous variables**

In credit scoring, continuous variables are also transformed using the WoE. To do
this, first variables are sorted into a discrete number of bins, and then these
bins are encoded with the WoE as explained here for categorical variables. You can
do this by combining the use of the equal width, equal frequency or arbitrary
discretisers.

More details
^^^^^^^^^^^^

Check also:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/WoEEncoder.ipynb>`_
