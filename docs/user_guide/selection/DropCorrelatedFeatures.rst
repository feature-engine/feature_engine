.. _drop_correlated:

.. currentmodule:: feature_engine.selection

DropCorrelatedFeatures
======================

The :class:`DropCorrelatedFeatures()` finds and removes correlated variables from a dataframe.
Correlation is calculated with `pandas.corr()`. All correlation methods supported by `pandas.corr()`
can be used in the selection, including Pearson's, Kendall, or Spearman. You can also pass a
bespoke correlation function, provided it returns a value between -1 and 1.

Features are removed on first found first removed basis, without any further insight. That is,
the first feature will be retained an all subsequent features that are correlated with it, will
be removed.

The transformer will examine all numerical variables automatically. Note that you could pass a
dataframe with categorical and datetime variables, and these will be ignored automatically.
Alternatively, you can pass a list with the variables you wish to evaluate.

Note: the order of the variables in the dataset matters! A different variable order can
lead to different results. To make the transformer consistent, use the `'order_by'`
parameter.

Example
-------

Let's create a toy dataframe where 4 of the features are correlated:

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from feature_engine.selection import DropCorrelatedFeatures

    # make dataframe with some correlated variables
    def make_data():
        X, y = make_classification(n_samples=1000,
                               n_features=12,
                               n_redundant=4,
                               n_clusters_per_class=1,
                               weights=[0.50],
                               class_sep=2,
                               random_state=1)

        # trasform arrays into pandas df and series
        colnames = ['var_'+str(i) for i in range(12)]
        X = pd.DataFrame(X, columns =colnames)
        return X

    X = make_data()

Now, we set up :class:`DropCorrelatedFeatures()` to find and remove variables which
(absolute) correlation coefficient is bigger than 0.8:

.. code:: python

    tr = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8)


With `fit()` the transformer finds the correlated variables and with `transform()` it drops
them from the dataset:

.. code:: python

    Xt = tr.fit_transform(X)

The correlated feature groups are stored in the transformer's attributes:

.. code:: python

    tr.correlated_feature_sets_


.. code:: python

    [{'var_0', 'var_8'}, {'var_4', 'var_6', 'var_7', 'var_9'}]


The features that will be removed from the dataset are also stored in an attribute:

..  code:: python

    tr.features_to_drop_

.. code:: python

    {'var_6', 'var_7', 'var_8', 'var_9'}

If we now go ahead and print the transformed data, we see that the correlated features
have been removed.

.. code:: python

    print(print(Xt.head()))

.. code:: python

              var_0     var_1     var_2     var_3     var_4     var_5    var_10  \
    0  1.471061 -2.376400 -0.247208  1.210290 -3.247521  0.091527  2.070526
    1  1.819196  1.969326 -0.126894  0.034598 -2.910112 -0.186802  1.184820
    2  1.625024  1.499174  0.334123 -2.233844 -3.399345 -0.313881 -0.066448
    3  1.939212  0.075341  1.627132  0.943132 -4.783124 -0.468041  0.713558
    4  1.579307  0.372213  0.338141  0.951526 -3.199285  0.729005  0.398790

         var_11
    0 -1.989335
    1 -1.309524
    2 -0.852703
    3  0.484649
    4 -0.186530

The variable order matters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's now compare how the order of the variables in the training dataset may affect the
result, that is, the selected features.

We'll start by reducing the number of unique values in our dataframe and repeat the
feature selection procedure, without altering the order of the variables in the training
set.

.. code:: python

    X = X.round(3)
    tr = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8)
    Xt = tr.fit_transform(X)
    tr.correlated_feature_sets_

We see in the following output that we obtain the same feature sets as previously:

..  code:: python

    [{'var_0', 'var_8'}, {'var_4', 'var_6', 'var_7', 'var_9'}]

And the transformer will remove the same variables as before:

..  code:: python

    tr.features_to_drop_

..  code:: python

    {'var_6', 'var_7', 'var_8', 'var_9'}

Now, let's order the variables by the number of unique values before the feature selection
process:

.. code:: python

    tr = DropCorrelatedFeatures(
        variables=None, method='pearson', threshold=0.8, order_by='unique')
    Xt = tr.fit_transform(X.round(3))
    tr.correlated_feature_sets_
    
Now, the transformer finds different correlation subsets:

..  code:: python
    
    [{'var_4', 'var_6', 'var_7', 'var_9'}, {'var_0', 'var_8'}]

The different order in the subsets will lead to a different feature selection subset:

..  code:: python

    tr.features_to_drop_

..  code:: python

    {'var_0', 'var_4', 'var_6', 'var_9'}
    

More details
^^^^^^^^^^^^

In this notebook, we show how to use :class:`DropCorrelatedFeatures()` with a different
relation metric:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/selection/Drop-Correlated-Features.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
