.. _smart_correlation:

.. currentmodule:: feature_engine.selection

SmartCorrelatedSelection
========================

When we have big datasets, more than 2 features can be correlated. We could have 3, 4 or
more features that are correlated. Thus, which one should be keep and which ones should we
drop?

:class:`SmartCorrelatedSelection` tries to answer this question.

From a group of correlated variables, the :class:`SmartCorrelatedSelection` will retain
the variable with:

- the highest variance
- the highest cardinality
- the least missing data
- the most important (based on single feature machine learning models)

And drop the rest.

Features with higher diversity of values (higher variance or cardinality), tend to be more
predictive, whereas features with least missing data, tend to be more useful.

Procedure
---------

:class:`SmartCorrelatedSelection` will first find correlated feature groups using any
correlation method supported by `pandas.corr()`, or a user defined function that returns
a value between -1 and 1.

Then, from each group of correlated features, it will try and identify the best candidate
based on the above criteria.

If the criteria is based on feature importance, :class:`SmartCorrelatedSelection` will
train a machine learning model using each one of the features in a correlated feature
group, calculate the models performance, and select the feature that returned the highest
performing model. In other words, it trains single feature models, and retains the
feature of the highest performing model.

If the criteria is based on variance or cardinality, :class:`SmartCorrelatedSelection` will
determine these attributes for each feature in the group and retain that one with the highest.

If the criteria is based on missing data, :class:`SmartCorrelatedSelection` will determine the
number of NA in each feature from the correlated group and keep the one with less NA.

**Example**

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

With `fit()` the transformer finds the correlated variables and selects the one to keep.
With `transform()` it drops them from the dataset:

.. code:: python

    Xt = tr.fit_transform(X)

The correlated feature groups are stored in the transformer's attributes:

.. code:: python

    tr.correlated_feature_sets_

Note that in the second group, 4 features are correlated among themselves.

.. code:: python

    [{'var_4', 'var_6', 'var_7', 'var_9'}, {'var_0', 'var_8'}]

We can identify from each group which feature will be retained and which ones removed
by inspecting the dictionary:

.. code:: python

    tr.correlated_feature_dict_

In the dictionary below we see that from the first correlated group, `var_7` is a key,
hence it will be retained, whereas variables 4, 6 and  9 are values, which means that
they are correlated to `var_7` and will therefore be removed.

.. code:: python

   {'var_7': {'var_4', 'var_6', 'var_9'}, 'var_8': {'var_0'}}

Similarly, `var_8` is a key and will be retained, whereas the `var_0` is a value, which
means that it was found correlated to `var_8` and will therefore be removed.

The features that will be removed from the dataset are stored in a different attribute
as well:

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



More details
^^^^^^^^^^^^

In this notebook, we show how to use :class:`SmartCorrelatedSelection` with a different
relation metric:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Smart-Correlation-Selection.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.

For more details about this and other feature selection methods check out these resources:

- `Feature selection for machine learning <https://www.trainindata.com/p/feature-selection-for-machine-learning>`_, online course.
- `Feature selection in machine learning <https://leanpub.com/feature-selection-in-machine-learning>`_, book.
