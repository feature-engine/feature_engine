SmartCorrelatedSelection
========================

API Reference
-------------

.. autoclass:: feature_engine.selection.SmartCorrelatedSelection
    :members:

Example
-------

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

        # trasform arrays into pandas df and series
        colnames = ['var_'+str(i) for i in range(12)]
        X = pd.DataFrame(X, columns=colnames)
        return X


    X = make_data()


    # set up the selector
    tr = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="variance",
        estimator=None,
    )

    Xt = tr.fit_transform(X)

    tr.correlated_feature_sets_


.. code:: python

    [{'var_0', 'var_8'}, {'var_4', 'var_6', 'var_7', 'var_9'}]

..  code:: python

    tr.selected_features_

.. code:: python

   ['var_1', 'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_8', 'var_7']

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
