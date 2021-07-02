LogCpTransformer
================

API Reference
-------------

.. autoclass:: feature_engine.transformation.LogCpTransformer
    :members:


Example
-------

.. code:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston

    from feature_engine import transformation as vt

    # Load dataset
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)

    # Separate into train and test sets
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=0)

    # set up the variable transformer
    tf = vt.LogCpTransformer(variables = [7, 12], C="auto")

    # fit the transformer
    tf.fit(X_train)

    # transform the data
    train_t= tf.transform(X_train)
    test_t= tf.transform(X_test)

    # learned constant C
    tf.C_
    >>>
    {7: 2.1742, 12: 2.73}

    # un-transformed variable
    X_train[12].hist()

.. image:: ../images/logcpraw.png

.. code:: python

    # transformed variable
    train_t[12].hist()

.. image:: ../images/logcptransform.png
