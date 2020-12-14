DropCorrelatedFeatures
======================

API Reference
-------------

.. autoclass:: feature_engine.selection.DropCorrelatedFeatures
    :members:

Example
-------

The DropCorrelatedFeatures() finds and removes correlated variables from a dataframe.
The user can pass a list of variables to examine, or alternatively the selector will
examine all numerical variables in the data set.

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

    tr = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8)

    Xt = tr.fit_transform(X)

    tr.correlated_feature_sets_


.. code:: python

    [{'var_0', 'var_8'}, {'var_4', 'var_6', 'var_7', 'var_9'}]

..  code:: python

    tr.correlated_features_

.. code:: python

    {'var_6', 'var_7', 'var_8', 'var_9'}

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


