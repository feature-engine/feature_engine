.. _smart_correlation:

.. currentmodule:: feature_engine.selection

SmartCorrelatedSelection
========================

When dealing with datasets containing numerous features, it's common for more than two
features to exhibit correlations with each other. This correlation might manifest among
three, four, or even more features within the dataset. Consequently, determining which
features to retain and which ones to eliminate becomes a crucial consideration.

Deciding which features to retain from a correlated group involves several strategies,
such us:

1. **Model Performance**: Some features returns model with higher performance than others.

2. **Variability and Cardinality**: Features with higher variability or cardinality often provide more information about the target variable.

3. **Missing Data**: Features with less missing data are generally more reliable and informative.

We can apply this selection strategies out of the box with the :class:`SmartCorrelatedSelection`.

From a group of correlated variables, the :class:`SmartCorrelatedSelection` will retain
the variable with:

- the highest variance
- the highest cardinality
- the least missing data
- the best performing model (based on a single feature)

The remaining features within each correlated group will be dropped.

Features with higher diversity of values (higher variance or cardinality), tend to be more
predictive, whereas features with least missing data, tend to be more useful.

Alternatively, directly training a model using each feature within the group and retaining
the one that trains the best performing model, directly evaluates the influence of the
feature on the target.

Procedure
---------

:class:`SmartCorrelatedSelection` first finds correlated feature groups using any
correlation method supported by `pandas.corr()`, or a user defined function that returns
a value between -1 and 1.

Then, from each group of correlated features, it will try and identify the best candidate
based on the above criteria.

If the criteria is based on model performance, :class:`SmartCorrelatedSelection` will
train a single feature machine learning model, using each one of the features in a correlated
group, calculate the model's performance, and select the feature that returned the highest
performing model. In simpler words, it trains single feature models, and retains the
feature of the highest performing model.

If the criteria is based on variance or cardinality, :class:`SmartCorrelatedSelection` will
determine these attributes for each feature in the group and retain that one with the highest.
Note however, that variability is dominated by the variable's scale. Hence, **variables with
larger scales will dominate the selection procedure**, unless you have a scaled dataset.

If the criteria is based on missing data, :class:`SmartCorrelatedSelection` will determine the
number of NA in each feature from the correlated group and keep the one with less NA.

Variance
~~~~~~~~

Let's see how to use :class:`SmartCorrelatedSelection` in a toy example. Let's create a
toy dataframe with 4 correlated features:

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from feature_engine.selection import SmartCorrelatedSelection

    # make dataframe with some correlated variables
    def make_data():
        X, y = make_classification(n_samples=1000,
                                   n_features=12,
                                   n_redundant=4,
                                   n_clusters_per_class=1,
                                   weights=[0.50],
                                   class_sep=2,
                                   random_state=1)

        # transform arrays into pandas df and series
        colnames = ['var_'+str(i) for i in range(12)]
        X = pd.DataFrame(X, columns=colnames)
        return X

    X = make_data()

Now, we set up :class:`SmartCorrelatedSelection` to find features groups which (absolute)
correlation coefficient is >0.8. From these groups, we want to retain the feature with
highest variance:

.. code:: python

    # set up the selector
    tr = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="variance",
        estimator=None,
    )

With `fit()`, the transformer finds the correlated variables and selects the ones to keep.
With `transform()`, it drops the remaining features in the correlated group from the dataset:

.. code:: python

    Xt = tr.fit_transform(X)

The correlated feature groups are stored in the one of the transformer's attributes:

.. code:: python

    tr.correlated_feature_sets_

In the first group, 4 features are correlated to at least one of them. In the second group,
2 features are correlated.

.. code:: python

    [{'var_4', 'var_6', 'var_7', 'var_9'}, {'var_0', 'var_8'}]

:class:`SmartCorrelatedSelection` picks a feature, and then determines the correlation
of other features in the dataframe to it. Hence, all features in a group will be correlated
to this one feature, but they may or may not be correlated to the other features within
the group, because correlation is not transitive.

This feature that was used in the assessment, was either the one with the higher variance,
higher cardinality or smaller number of missing data. Or, if model performance was selected,
it was the one that came first in alphabetic order.

We can identify from each group which feature will be retained and which ones removed
by inspecting the following attribute:

.. code:: python

    tr.correlated_feature_dict_

In the dictionary below we see that from the first correlated group, `var_7` is a key,
hence it will be retained, whereas variables 4, 6 and  9 are values, which means that
they are correlated to `var_7` and will therefore be removed.

Because we are selecting features based on variability, `var_7` has the higher variability
from the group.

.. code:: python

   {'var_7': {'var_4', 'var_6', 'var_9'}, 'var_8': {'var_0'}}

Similarly, `var_8` is a key and will be retained, whereas the `var_0` is a value, which
means that it was found correlated to `var_8` and will therefore be removed.

We can corroborate that, for example, `var_7` had the highest variability as follows:

.. code:: python

    X[list(tr.correlated_feature_sets_[0])].std()

That command returns the following output, where we see that the variability of `var_7`
is the highest:

.. code:: python

    var_4    1.810273
    var_7    2.159634
    var_9    1.764249
    var_6    2.032947
    dtype: float64

The features that will be removed from the dataset are stored in the following attribute:

..  code:: python

    tr.features_to_drop_

.. code:: python

   ['var_6', 'var_4', 'var_9', 'var_0']

If we now go ahead and print the transformed data, we see that the correlated features
have been removed.

.. code:: python

    print(Xt.head())

.. code:: python

          var_1     var_2     var_3     var_5     var_7     var_8    var_10  \
    0 -2.376400 -0.247208  1.210290  0.091527 -2.230170  2.070483  2.070526
    1  1.969326 -0.126894  0.034598 -0.186802 -1.447490  2.421477  1.184820
    2  1.499174  0.334123 -2.233844 -0.313881 -2.240741  2.263546 -0.066448
    3  0.075341  1.627132  0.943132 -0.468041 -3.534861  2.792500  0.713558
    4  0.372213  0.338141  0.951526  0.729005 -2.053965  2.186741  0.398790

         var_11
    0 -1.989335
    1 -1.309524
    2 -0.852703
    3  0.484649
    4 -0.186530


Performance
~~~~~~~~~~~

Let's now select the feature that returns a machine learning model with the highest
performance, from each group. We'll use a decision tree.

We start by creating a toy dataframe:



.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier
    from feature_engine.selection import SmartCorrelatedSelection

    # make dataframe with some correlated variables
    def make_data():
        X, y = make_classification(n_samples=1000,
                                   n_features=12,
                                   n_redundant=4,
                                   n_clusters_per_class=1,
                                   weights=[0.50],
                                   class_sep=2,
                                   random_state=1)

        # transform arrays into pandas df and series
        colnames = ['var_'+str(i) for i in range(12)]
        X = pd.DataFrame(X, columns=colnames)
        return X, y

    X, y = make_data()

Let's now set up the selector:

.. code:: python

    tr = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=DecisionTreeClassifier(random_state=1),
        scoring='roc_auc',
        cv=3,
    )

Next, we fit the selector to the data. Here, as we are training a model, we also need to
pass the target variable:

.. code:: python

    Xt = tr.fit_transform(X, y)

Let's explore the correlated feature groups:

.. code:: python

    tr.correlated_feature_sets_

We see that the groups of correlated features are slightly different, because in this
cases, the features were assessed in alphabetical order, whereas when we used the variance
the features we sorted based on their standard deviation for the assessment.

.. code:: python

    [{'var_0', 'var_8'}, {'var_4', 'var_6', 'var_7', 'var_9'}]

We can find the feature that will be retained as the key in the following attribute:

.. code:: python

    tr.correlated_feature_dict_

The variables `var_0` and `var_7` will be retained, and the remaining ones will be dropped.

.. code:: python

    {'var_0': {'var_8'}, 'var_7': {'var_4', 'var_6', 'var_9'}}

We find the variables that will be dropped in the following attribute:

.. code:: python

    tr.features_to_drop_

.. code:: python

    ['var_8', 'var_4', 'var_6', 'var_9']

And now we can print the resulting dataframe after the transformation:

.. code:: python

    print(Xt.head())

.. code:: python

          var_0     var_1     var_2     var_3     var_5     var_7    var_10  \
    0  1.471061 -2.376400 -0.247208  1.210290  0.091527 -2.230170  2.070526
    1  1.819196  1.969326 -0.126894  0.034598 -0.186802 -1.447490  1.184820
    2  1.625024  1.499174  0.334123 -2.233844 -0.313881 -2.240741 -0.066448
    3  1.939212  0.075341  1.627132  0.943132 -0.468041 -3.534861  0.713558
    4  1.579307  0.372213  0.338141  0.951526  0.729005 -2.053965  0.398790

         var_11
    0 -1.989335
    1 -1.309524
    2 -0.852703
    3  0.484649
    4 -0.186530

Let's examine other attributes that may be useful. Like with any Scikit-learn transformer
we can obtain the names of the features in the resulting dataframe as follows:

.. code:: python

    tr.get_feature_names_out()

.. code:: python

    ['var_0', 'var_1', 'var_2', 'var_3', 'var_5', 'var_7', 'var_10', 'var_11']

We also find the `get_support` method that flags the features that will be retained from
the dataframe:

.. code:: python

    tr.get_support()

.. code:: python

    [True, True, True, True, False, True, False, True, False, False, True, True]

And that's it!

Additional resources
--------------------

In this notebook, we show how to use :class:`SmartCorrelatedSelection` with a different
relation metric:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Smart-Correlation-Selection.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.

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
   :target: https://www.trainindata.com/p/feature-selection-in-machine-learning-book

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