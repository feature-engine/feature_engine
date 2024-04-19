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
    y = pd.Series(diabetes_y)

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

    {0: -0.0035681361984126747,
     1: 0.041170843574652394,
     2: 0.1920054944393057,
     3: 0.07007527443645178,
     4: 0.49871458125373913,
     5: 0.1802858704499694,
     6: 0.025536233845966705,
     7: 0.024058931694668884,
     8: 0.40901959802129045,
     9: 0.004487448637912506}

:class:`SelectByShuffling()` also stores the features that will be dropped based on the
threshold indicated.

.. code:: python

    tr.features_to_drop_

.. code:: python

    [0, 1, 3, 6, 7, 9]

If we now print the transformed data, we see that the features above were removed.

..  code:: python

    print(Xt.head())

..  code:: python

              2         4         5         8
    0  0.061696 -0.044223 -0.034821  0.019907
    1 -0.051474 -0.008449 -0.019163 -0.068332
    2  0.044451 -0.045599 -0.034194  0.002861
    3 -0.011595  0.012191  0.024991  0.022688
    4 -0.036385  0.003935  0.015596 -0.031988
    

Additional resources
--------------------

For more details about this and other feature selection methods check out these resources:

.. figure::  ../../images/fsml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-selection-for-machine-learning

   Feature Selection for Machine Learning

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

.. figure::  ../../images/fsmlbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://leanpub.com/feature-selection-in-machine-learning

   Feature Selection in Machine Learning

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
|

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.