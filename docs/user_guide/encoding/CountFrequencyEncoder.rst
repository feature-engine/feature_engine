.. _count_freq_encoder:

.. currentmodule:: feature_engine.encoding

CountFrequencyEncoder
=====================

The :class:`CountFrequencyEncoder()` replaces categories by either the count or the
percentage of observations per category. For example in the variable colour, if 10
observations are blue, blue will be replaced by 10. Alternatively, if 10% of the
observations are blue, blue will be replaced by 0.1.

Let's look at an example using the Titanic Dataset.

First, let's load the data and separate it into train and test:

.. code:: python

    from sklearn.model_selection import train_test_split
    from feature_engine.datasets import load_titanic
    from feature_engine.encoding import CountFrequencyEncoder

    X, y = load_titanic(
        return_X_y_frame=True,
        handle_missing=True,
        predictors_only=True,
        cabin="letter_only",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )

    print(X_train.head())

We see the resulting data below:

.. code:: python

          pclass     sex        age  sibsp  parch     fare cabin embarked
    501        2  female  13.000000      0      1  19.5000     M        S
    588        2  female   4.000000      1      1  23.0000     M        S
    402        2  female  30.000000      1      0  13.8583     M        C
    1193       3    male  29.881135      0      0   7.7250     M        Q
    686        3  female  22.000000      0      0   7.7250     M        Q

Now, we set up the :class:`CountFrequencyEncoder()` to replace the categories by their
frequencies, only in the 3 indicated variables:

.. code:: python

    encoder = CountFrequencyEncoder(
    encoding_method='frequency',
    variables=['cabin', 'pclass', 'embarked'],
    ignore_format=True,
    )

    encoder.fit(X_train)

With `fit()` the encoder learns the frequencies of each category, which are stored in
its `encoder_dict_` parameter:

.. code:: python

	encoder.encoder_dict_

In the `encoder_dict_` we find the frequencies for each one of the categories of each
variable that we want to encode. This way, we can map the original value to the new
value.

.. code:: python

	{'cabin': {'M': 0.7663755458515283,
		'C': 0.07751091703056769,
		'B': 0.04585152838427948,
		'E': 0.034934497816593885,
		'D': 0.034934497816593885,
		'A': 0.018558951965065504,
		'F': 0.016375545851528384,
		'G': 0.004366812227074236,
		'T': 0.001091703056768559},
	'pclass': {3: 0.5436681222707423,
		1: 0.25109170305676853,
		2: 0.2052401746724891},
	'embarked': {'S': 0.7117903930131004,
		'C': 0.19541484716157206,
		'Q': 0.0906113537117904,
		'Missing': 0.002183406113537118}}

We can now go ahead and replace the original strings with the numbers:

.. code:: python

    train_t = encoder.transform(X_train)
    test_t = encoder.transform(X_test)

    print(train_t.head())

Below we see the that the original variables were replaced with the frequencies:

.. code:: python

            pclass     sex        age  sibsp  parch     fare     cabin  embarked
    501   0.205240  female  13.000000      0      1  19.5000  0.766376  0.711790
    588   0.205240  female   4.000000      1      1  23.0000  0.766376  0.711790
    402   0.205240  female  30.000000      1      0  13.8583  0.766376  0.195415
    1193  0.543668    male  29.881135      0      0   7.7250  0.766376  0.090611
    686   0.543668  female  22.000000      0      0   7.7250  0.766376  0.090611

Additional resources
--------------------

In the following notebook, you can find more details into the :class:`CountFrequencyEncoder()`
functionality and example plots with the encoded variables:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/encoding/CountFrequencyEncoder.ipynb>`_

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