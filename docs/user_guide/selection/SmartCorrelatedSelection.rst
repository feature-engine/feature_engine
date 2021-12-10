.. _smart_correlation:

.. currentmodule:: feature_engine.selection

SmartCorrelatedSelection
========================

When we have big datasets, more than 2 features can be correlated. We could have 3, 4 or
more features that are correlated. Thus, which one should be keep and which ones should we
drop?

:class:`SmartCorrelatedSelection` tries to answer this question.

From a group of correlated variables, the :class:`SmartCorrelatedSelection` will retain
the one with:

- the highest variance
- the highest cardinality
- the least missing data
- the most important (based on embedded selection methods)

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
train a machine learning model using the correlated feature group, derive the feature importance
from this model, end then keep the feature with the highest important.

:class:`SmartCorrelatedSelection` works with machine learning models that derive coefficients
or feature importance values.

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

    [{'var_0', 'var_8'}, {'var_4', 'var_6', 'var_7', 'var_9'}]

In the following attribute we find the features that will be removed from the dataset:

..  code:: python

    tr.features_to_drop_

.. code:: python

   ['var_0', 'var_4', 'var_6', 'var_9']

If we now go ahead and print the transformed data, we see that the correlated features
have been removed.

.. code:: python

    print(print(Xt.head()))

.. code:: python

          var_1     var_2     var_3     var_5    var_10    var_11     var_8  \
    0 -2.376400 -0.247208  1.210290  0.091527  2.070526 -1.989335  2.070483
    1  1.969326 -0.126894  0.034598 -0.186802  1.184820 -1.309524  2.421477
    2  1.499174  0.334123 -2.233844 -0.313881 -0.066448 -0.852703  2.263546
    3  0.075341  1.627132  0.943132 -0.468041  0.713558  0.484649  2.792500
    4  0.372213  0.338141  0.951526  0.729005  0.398790 -0.186530  2.186741

          var_7
    0 -2.230170
    1 -1.447490
    2 -2.240741
    3 -3.534861
    4 -2.053965


More details
^^^^^^^^^^^^

In this notebook, we show how to use :class:`SmartCorrelatedSelection` with a different
relation metric:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Smart-Correlation-Selection.ipynb>`_

