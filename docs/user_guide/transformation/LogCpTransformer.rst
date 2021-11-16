.. _log_cp:

.. currentmodule:: feature_engine.transformation

LogCpTransformer
================

The :class:`LogCpTransformer()` applies the transformation log(x + C), where C is a
positive constant.

You can enter the positive quantity to add to the variable. Alternatively, the transformer
will find the necessary quantity to make all values of the variable positive.

**Example**

Let's load the boston house prices dataset that comes baked into Scikit-learn and
separate it into train and test sets.

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


Now we want to apply the logarithm to 2 of the variables in the dataset using the
:class:`LogCpTransformer()`. We want the transformer to detect automatically the
quantity "C" that needs to be added to the variable:

.. code:: python

    # set up the variable transformer
    tf = vt.LogCpTransformer(variables = [7, 12], C="auto")

    # fit the transformer
    tf.fit(X_train)

With `fit()` the :class:`LogCpTransformer()` learns the quantity "C" and stores it as
an attribute. We can visualise the learned parameters as follows:

.. code:: python

    # learned constant C
    tf.C_

.. code:: python

    {7: 2.1742, 12: 2.73}

We can now go ahead and transform the variables:

.. code:: python

    # transform the data
    train_t= tf.transform(X_train)
    test_t= tf.transform(X_test)

Then we can plot the original variable distribution:

.. code:: python

    # un-transformed variable
    X_train[12].hist()

.. image:: ../../images/logcpraw.png

And the distribution of the transformed variable:

.. code:: python

    # transformed variable
    train_t[12].hist()

.. image:: ../../images/logcptransform.png

More details
^^^^^^^^^^^^

You can find more details about the :class:`LogCpTransformer()` here:

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/transformation/LogCpTransformer.ipynb>`_

All notebooks can be found in a `dedicated repository <https://github.com/feature-engine/feature-engine-examples>`_.
