.. _feature_shuffling:

.. currentmodule:: feature_engine.selection

SelectByShuffling
=================

The :class:`SelectByShuffling()` selects important features if a random permutation of
their values decreases the model performance. If the feature is predictive, a random
shuffle of the values across the rows, should return predictions that are off the truth.
If the feature is not predictive, their values should have a minimal impact on the prediction.

Procedure
---------

The algorithm works as follows:

1. Train a machine learning model using all features
2. Determine a model performance metric of choice
3. Shuffle the order of 1 feature values
4. Use the model trained in 1 to obtain new predictions
5. Determine the performance with the predictions in 4
6. If there is a drop in performance beyond a threshold, keep the feature.
7. Repeat 3-6 until all features are examined.

**Example**

Let's see how to use this transformer with the diabetes dataset that comes in Scikit-learn.
First, we load the data:

.. code:: python

    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from feature_engine.selection import SelectByShuffling

    # load dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.DataFrame(diabetes_y)

Now, we set up the model for which we want to have the performance drop evaluated:

.. code:: python

    # initialize linear regresion estimator
    linear_model = LinearRegression()

Now, we instantiate :class:`SelectByShuffling()` to select features by shuffling, based on
the r2 of the model from the previous cell, using 3 fold cross-validation. The parameter
`threshold` was left to None, which means that features will be selected if the performance
drop is bigger than the mean drop caused by all features.

.. code:: python

    # initialize feature selector
    tr = SelectByShuffling(estimator=linear_model, scoring="r2", cv=3)


With `fit()` the transformer finds the important variables, that is, those which values
permutations caused a drop in the model performance. With `transform()` it drops them
from the dataset:

.. code:: python

    # fit transformer
    Xt = tr.fit_transform(X, y)

:class:`SelectByShuffling()` stores the performance of the model trained using all the features
in its attribute:

.. code:: python

    tr.initial_model_performance_

.. code:: python

    0.488702767247119

:class:`SelectByShuffling()` also stores the performance change caused by every single
feature after shuffling. In case you are not satisfied with the threshold used, you can get
an idea of where the threshold could be by looking at these values:

..  code:: python

    tr.performance_drifts_

.. code:: python

    {0: -0.02368121940502793,
     1: 0.017909161264480666,
     2: 0.18565460365508413,
     3: 0.07655405817715671,
     4: 0.4327180164470878,
     5: 0.16394693824418372,
     6: -0.012876023845921625,
     7: 0.01048781540981647,
     8: 0.3921465005640224,
     9: -0.01427065640301245}

:class:`SelectByShuffling()` also stores the features that will be dropped based on the
threshold indicated.

.. code:: python

    tr.features_to_drop_

.. code:: python

    [0, 1, 3, 6, 7, 9]

