.. _dtree_features:

.. currentmodule:: feature_engine.creation

DecisionTreeFeatures
====================

The winners of the KDD 2009 competition observed that many features had high
mutual information with the target, but low correlation, leading them to conclude
that the relationships were non-linear. While non-linear relationships can be
captured by non-linear models, to leverage the information from these features with
linear models, we need to somehow transform that information into a linear, or
monotonic relationship with the target.

The output of decision trees, that is, their predictions, should be monotonic with
the target, if there is a good fit for the tree.

In addition, decision trees trained on 2 or more features could capture feature
interactions that simpler models would miss.

By enriching the dataset with features resulting from the predictions of decision trees,
we can create better performing models. On the downside the features resulting
from decision trees, are not easy to interpret or explain.

:class:`DecisionTreeFeatures()` creates and adds features resulting from the predictions
of decision trees trained on 1 or more features.

Values of the tree based features
---------------------------------

If we create features for regression, :class:`DecisionTreeFeatures()` will train scikit-learn's
`DecisionTreeRegressor` under the hood, and the features are derived from the `predict` method
of these regressors. Hence, the features will be in the scale of the target. Remember however,
that the output of decision tree regressors is not continuous, it is a piecewise function.

If we create features for classification, :class:`DecisionTreeFeatures()` will train scikit-learn's
`DecisionTreeClassifier` under the hood. If the target is binary, the resulting features
are the output of the `predict_proba` method of the model corresponding to the predictions
of class 1. If the output is multiclass, on the other hand, the features are derived from
the `predict` method, and hence return the predicted class.

Examples
--------

In the rest of the document, we'll show the versatility of :class:`DecisionTreeFeatures()`
to create multiple features by using decision trees.

Let's start by loading and displaying the California housing dataset from sklearn

.. code:: python

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from feature_engine.creation import DecisionTreeFeatures

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True)
    print(X.head())

In the following output we see the dataframe:

.. code:: python

       MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup
    0  8.3252      41.0  6.984127   1.023810       322.0  2.555556
    1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842
    2  7.2574      52.0  8.288136   1.073446       496.0  2.802260
    3  5.6431      52.0  5.817352   1.073059       558.0  2.547945
    4  3.8462      52.0  6.281853   1.081081       565.0  2.181467

Let's split the dataset into a training and a testing set:

.. code:: python

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

Combining features - integers
-----------------------------

We'll set up :class:`DecisionTreeFeatures()` to create **all possible** combinations of 2
features. To create all possible combinations we use integers with the `features_to_combine`
parameter:

.. code:: python

    dtf = DecisionTreeFeatures(features_to_combine=2)
    dtf.fit(X_train, y_train)

If we leave the parameter `variables` to `None`, :class:`DecisionTreeFeatures()` will combine
all numerical variables in the training set, in the way we indicate in `features_to_combine`.
Since we set `features_to_combine=2`, the transformer will create all possible combinations of
1 or 2 variables.

We can find the feature combinations that will be used to train the trees as follows:

.. code:: python

    dtf.input_features_

In the following output we see the combinations of 1 and 2 features that will be used
to train decision trees, based of all the numerical variables in the training set:

.. code:: python

    ['MedInc',
     'HouseAge',
     'AveRooms',
     'AveBedrms',
     'Population',
     'AveOccup',
     ['MedInc', 'HouseAge'],
     ['MedInc', 'AveRooms'],
     ['MedInc', 'AveBedrms'],
     ['MedInc', 'Population'],
     ['MedInc', 'AveOccup'],
     ['HouseAge', 'AveRooms'],
     ['HouseAge', 'AveBedrms'],
     ['HouseAge', 'Population'],
     ['HouseAge', 'AveOccup'],
     ['AveRooms', 'AveBedrms'],
     ['AveRooms', 'Population'],
     ['AveRooms', 'AveOccup'],
     ['AveBedrms', 'Population'],
     ['AveBedrms', 'AveOccup'],
     ['Population', 'AveOccup']]

Let's now add the new features to the data:

.. code:: python

    train_t = dtf.transform(X_train)
    test_t = dtf.transform(X_test)

    print(test_t.head())

In the following output we see the resulting data with the features derived from
decision trees:

.. code:: python

           MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  \
    14740  4.1518      22.0  5.663073   1.075472      1551.0  4.180593
    10101  5.7796      32.0  6.107226   0.927739      1296.0  3.020979
    20566  4.3487      29.0  5.930712   1.026217      1554.0  2.910112
    2670   2.4511      37.0  4.992958   1.316901       390.0  2.746479
    15709  5.0049      25.0  4.319261   1.039578       649.0  1.712401

           tree(MedInc)  tree(HouseAge)  tree(AveRooms)  tree(AveBedrms)  ...  \
    14740      2.204822        2.130618        2.001950         2.080254  ...
    10101      2.975513        2.051980        2.001950         2.165554  ...
    20566      2.204822        2.051980        2.001950         2.165554  ...
    2670       1.416771        2.051980        1.802158         1.882763  ...
    15709      2.420124        2.130618        1.802158         2.165554  ...

           tree(['HouseAge', 'AveRooms'])  tree(['HouseAge', 'AveBedrms'])  \
    14740                        1.885406                         2.124812
    10101                        1.885406                         2.124812
    20566                        1.885406                         2.124812
    2670                         1.797902                         1.836498
    15709                        1.797902                         2.124812

           tree(['HouseAge', 'Population'])  tree(['HouseAge', 'AveOccup'])  \
    14740                          2.004703                        1.437440
    10101                          2.004703                        2.257968
    20566                          2.004703                        2.257968
    2670                           2.123579                        2.257968
    15709                          2.123579                        2.603372

           tree(['AveRooms', 'AveBedrms'])  tree(['AveRooms', 'Population'])  \
    14740                         2.099977                          1.878989
    10101                         2.438937                          2.077321
    20566                         2.099977                          1.878989
    2670                          1.728401                          1.843904
    15709                         1.821467                          1.843904

           tree(['AveRooms', 'AveOccup'])  tree(['AveBedrms', 'Population'])  \
    14740                        1.719582                           2.056003
    10101                        2.156884                           2.056003
    20566                        2.156884                           2.056003
    2670                         1.747990                           1.882763
    15709                        2.783690                           2.221092

           tree(['AveBedrms', 'AveOccup'])  tree(['Population', 'AveOccup'])
    14740                         1.400491                          1.484939
    10101                         2.153210                          2.059187
    20566                         2.153210                          2.059187
    2670                          1.861020                          2.235743
    15709                         2.727460                          2.747390

    [5 rows x 27 columns]

Combining features - Lists
--------------------------

Let's say that we want to create features based of trees trained of 2 or more variables. Instead of using
an integer in `features_to_combine`, we need to pass a list of integers, telling :class:`DecisionTreeFeatures()`
to make all possible combinations of the integers mentioned in the list.

We'll set up the transformer to create features based on all possible combinations of
2 and 3 features of just 3 of the numerical variables this time:

.. code:: python

    dtf = DecisionTreeFeatures(
        variables=["AveRooms", "AveBedrms", "Population"],
        features_to_combine=[2,3])

    dtf.fit(X_train, y_train)

If we now examine the feature combinations:

.. code:: python

    dtf.input_features_

We see that they are based of combinations of 2 or 3 of the variables that we set in
the `variables` parameter:

.. code:: python

    [['AveRooms', 'AveBedrms'],
     ['AveRooms', 'Population'],
     ['AveBedrms', 'Population'],
     ['AveRooms', 'AveBedrms', 'Population']]

We can now add the features to the data and inspect the result:

.. code:: python

    train_t = dtf.transform(X_train)
    test_t = dtf.transform(X_test)

    print(test_t.head())

In the following output we see the dataframe with the new features:

.. code:: python

           MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  \
    14740  4.1518      22.0  5.663073   1.075472      1551.0  4.180593
    10101  5.7796      32.0  6.107226   0.927739      1296.0  3.020979
    20566  4.3487      29.0  5.930712   1.026217      1554.0  2.910112
    2670   2.4511      37.0  4.992958   1.316901       390.0  2.746479
    15709  5.0049      25.0  4.319261   1.039578       649.0  1.712401

           tree(['AveRooms', 'AveBedrms'])  tree(['AveRooms', 'Population'])  \
    14740                         2.099977                          1.878989
    10101                         2.438937                          2.077321
    20566                         2.099977                          1.878989
    2670                          1.728401                          1.843904
    15709                         1.821467                          1.843904

           tree(['AveBedrms', 'Population'])  \
    14740                           2.056003
    10101                           2.056003
    20566                           2.056003
    2670                            1.882763
    15709                           2.221092

           tree(['AveRooms', 'AveBedrms', 'Population'])
    14740                                       2.099977
    10101                                       2.438937
    20566                                       2.099977
    2670                                        1.843904
    15709                                       1.843904

Specifying the feature combinations - tuples
--------------------------------------------

We can indicate precisely the features that we want to use as input of the decision trees.
Let's make a tuple containing the features combinations. We want a tree trained with
Population, a tree trained with Population and AveOccup, and a tree trained with
those 2 variables plus HouseAge:

.. code:: python

    features = (('Population'), ('Population', 'AveOccup'),
                ('Population', 'AveOccup', 'HouseAge'))

Now, we pass this tuple to :class:`DecisionTreeFeatures()` and note that we can leave
the parameter `variables` to the default, because with tuples, that parameter is
ignored:

.. code:: python

    dtf = DecisionTreeFeatures(
        variables=None,
        features_to_combine=features,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    dtf.fit(X_train, y_train)

If we inspect the input features, it will coincide with the tuple we passed to
`features_to_combine`:

.. code:: python

    dtf.input_features_

We see that the input features are those from the tuple:

.. code:: python

    ['Population',
     ['Population', 'AveOccup'],
     ['Population', 'AveOccup', 'HouseAge']]

And now we can go ahead and add the features to the data:

.. code:: python

    train_t = dtf.transform(X_train)
    test_t = dtf.transform(X_test)

Examining the new features
--------------------------

:class:`DecisionTreeFeatures()` appends the word `tree` to the new features, so if
we wanted to display only the new features, we can do so as follows

.. code:: python

    tree_features = [var for var in test_t.columns if "tree" in var]
    print(test_t[tree_features].head())

.. code:: python

           tree(Population)  tree(['Population', 'AveOccup'])  \
    14740          2.008283                          1.484939
    10101          2.008283                          2.059187
    20566          2.008283                          2.059187
    2670           2.128961                          2.235743
    15709          2.128961                          2.747390

           tree(['Population', 'AveOccup', 'HouseAge'])
    14740                                      1.443097
    10101                                      2.257968
    20566                                      2.257968
    2670                                       2.257968
    15709                                      3.111251


Evaluating individual trees
---------------------------

We can evaluate the performance of each of the trees used to create the features, if
we so wish. Let's set up the :class:`DecisionTreeFeatures()`:

.. code:: python

    dtf = DecisionTreeFeatures(features_to_combine=2)
    dtf.fit(X_train, y_train)

:class:`DecisionTreeFeatures()` trains each tree with cross-validation. If we do not
pass a grid with hyperparameters, it will optimize the depth by default. We can find
the trained estimators like this:

.. code:: python

    dtf.estimators_

Because the estimators are trained with sklearn's `GridSearchCV`, what is stored is
the result of the search:

.. code:: python

    [GridSearchCV(cv=3, estimator=DecisionTreeRegressor(random_state=0),
                  param_grid={'max_depth': [1, 2, 3, 4]},
                  scoring='neg_mean_squared_error'),
     GridSearchCV(cv=3, estimator=DecisionTreeRegressor(random_state=0),
                  param_grid={'max_depth': [1, 2, 3, 4]},
                  scoring='neg_mean_squared_error'),
     ...

     GridSearchCV(cv=3, estimator=DecisionTreeRegressor(random_state=0),
                  param_grid={'max_depth': [1, 2, 3, 4]},
                  scoring='neg_mean_squared_error'),
     GridSearchCV(cv=3, estimator=DecisionTreeRegressor(random_state=0),
                  param_grid={'max_depth': [1, 2, 3, 4]},
                  scoring='neg_mean_squared_error')]

If you want to inspect an individual tree and it's performance, you can do so like this:

.. code:: python

    tree = dtf.estimators_[4]
    tree.best_params_

In the following output, we see the best parameters obtained for a tree trained based
of the feature **Population** to predict house price:

.. code:: python

    {'max_depth': 2}

If we want to check out the performance of the best tree during found in the grid search,
we can do so like this:

.. code:: python

    tree.score(X_test[['Population']], y_test)

The following performance value corresponds to the negative of the mean squared error
which is the metric optimised durign the search (you can select the metric to optimize
through the `scoring` parameter of :class:`DecisionTreeFeatures()`).

.. code:: python

    -1.3308515769033213

Note that you can also isolate the tree, and then obtain a performance metric:

.. code:: python

    tree.best_estimator_.score(X_test[['Population']], y_test)

In this case, the following performance metric corresponds to the R2, which is the
default metric returned by scikit-learn's DecisionTreeRegressor.

.. code:: python

    0.0017890442253447603

Dropping the original variables
-------------------------------

With :class:`DecisionTreeFeatures()`, we can automatically remove from the resulting
dataframe the features used as input from the decision trees. We need to set `drop_original`
to `True`.

.. code:: python

    dtf = DecisionTreeFeatures(
        variables=["AveRooms", "AveBedrms", "Population"],
        features_to_combine=[2,3],
        drop_original=True
    )

    dtf.fit(X_train, y_train)

    train_t = dtf.transform(X_train)
    test_t = dtf.transform(X_test)

    print(test_t.head())

We see in the resulting dataframe that the variables ["AveRooms", "AveBedrms", "Population"]
are not there:

.. code:: python

           MedInc  HouseAge  AveOccup  tree(['AveRooms', 'AveBedrms'])  \
    14740  4.1518      22.0  4.180593                         2.099977
    10101  5.7796      32.0  3.020979                         2.438937
    20566  4.3487      29.0  2.910112                         2.099977
    2670   2.4511      37.0  2.746479                         1.728401
    15709  5.0049      25.0  1.712401                         1.821467

           tree(['AveRooms', 'Population'])  tree(['AveBedrms', 'Population'])  \
    14740                          1.878989                           2.056003
    10101                          2.077321                           2.056003
    20566                          1.878989                           2.056003
    2670                           1.843904                           1.882763
    15709                          1.843904                           2.221092

           tree(['AveRooms', 'AveBedrms', 'Population'])
    14740                                       2.099977
    10101                                       2.438937
    20566                                       2.099977
    2670                                        1.843904
    15709                                       1.843904

Creating features for classification
------------------------------------

If we are creating features for a classifier instead of a regressor, the procedure is
identical. We just need to set the parameter `regression` to False.

Note that if you are creating features for binary classification, the added features
will contain the probabilities of class 1. If you are creating features for multi-class
classification, on the other hand, the features will contain the prediction of the class.


Additional resources
--------------------

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
   :target: https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587

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